import sys
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model
import torch
from torch.utils.data import Dataset
import transformers
from transformers import (
    AutoModelForCausalLM,
    default_data_collator,
    set_seed,
)
import time


def main(args):
    # Skip model initilization

    # deepspeed.ops.op_builder.CPUAdamBuilder().load()

    transformers.PreTrainedModel._initialize_weights = lambda x, *args, **kwargs: x
    torch.nn.init.normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.xavier_uniform_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_normal_ = lambda x, *args, **kwargs: x
    torch.nn.init.kaiming_uniform_ = lambda x, *args, **kwargs: x

    accelerator = Accelerator()
    model_name_or_path = args.model_name_or_path
    print("accelerator device:", accelerator.device)

    if accelerator.is_main_process:
        # is_local_main_process
        name = f"{args.gpu=},{args.batch=},{args.seq_len=}"
        wandb.init(
            project=f"{args.wb_project_prefix}_{args.seed}",
            name=name,
            config={"command": sys.argv, **vars(args)},
        )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    lr = 1e-5
    num_epochs = args.epochs
    batch_size = args.batch

    seed = args.seed
    set_seed(seed)

    class CustomDataset(Dataset):
        def __init__(self, num_rows, input_size):
            # 假设每个特征都是随机生成的
            self.input_ids = torch.randint(
                low=0, high=1024, size=(num_rows, input_size)
            )

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
            }

    # 使用示例
    num_rows = 192  # 可以根据需要调整数据集大小
    input_size = args.seq_len  # 可以根据需要调整输入大小
    train_dataset = CustomDataset(num_rows=num_rows, input_size=input_size)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size,
        pin_memory=True,
    )

    print("train_dataloader:", len(train_dataloader))

    # creating model
    if args.use_flash_attn:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
        )

    if args.use_better_transformer:
        model.to_bettertransformer()
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    iter = 0
    # new_epochs = 100

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # lr scheduler
    # lr_scheduler = get_linear_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=(len(train_dataloader) * num_epochs),
    # )

    (
        model,
        train_dataloader,
        optimizer,
        # lr_scheduler,
    ) = accelerator.prepare(
        model,
        train_dataloader,
        optimizer,
        # lr_scheduler,
    )
    # accelerator.print(model)

    start_time = time.time()
    for epoch in range(num_epochs):
        # with TorchTracemalloc() as tracemalloc:
        model.train()
        # total_loss = 0
        # print(len(train_dataloader))
        if iter >= args.max_iter:
            break

        print("len(train_dataloader) after:", len(train_dataloader))
        "len_dataloader * batchsize * gpu >= num_examples"
        for step, batch in enumerate(
            tqdm(train_dataloader, disable=not accelerator.is_main_process)
        ):
            if iter >= args.max_iter:
                break
            iter += 1

            input = batch["input_ids"]
            print(input.shape)

            accelerator.wait_for_everyone()
            t0 = time.time()

            # pre_mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            # pre_mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)

            outputs = model(input_ids=input, labels=input, use_cache=False)
            loss = outputs.loss

            accelerator.wait_for_everyone()
            t1 = time.time()

            # peak_mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)
            # peak_mem_reserved = torch.cuda.memory_reserved() / (1024 * 1024 * 1024)

            # total_loss += loss.detach().float()
            accelerator.backward(loss)

            accelerator.wait_for_everyone()
            t2 = time.time()

            optimizer.step()

            accelerator.wait_for_everyone()
            t3 = time.time()

            optimizer.zero_grad()

            # if accelerator.device == "cuda:0":
            if accelerator.is_main_process:
                accelerator.print(f"{accelerator.device=}, {step=}, {t3-t0=} ")
                wandb.log(
                    {
                        # "tput global(bs*seqlen/step_time)": (
                        #     8 * batch_size * input_size
                        # )
                        # / (t3 - t0),
                        # "tput per gpu(bs*seqlen/step_time)": (batch_size * input_size)
                        # / (t3 - t0),
                        "tput global(bs*seqlen/iter_time)": (
                            8 * batch_size * input_size
                        )
                        / (t2 - t0),
                        "tput per gpu(bs*seqlen/iter_time)": (batch_size * input_size)
                        / (t2 - t0),
                        "forward_time": t1 - t0,
                        "backward_time": t2 - t1,
                        "iteration time": t2 - t0,
                        # "step_time": t3 - t0,
                    },
                    step=iter,
                )

            """
            iteration:一般翻译为“迭代”,多数情况下就表示在训练过程中经过一个step的操作。
            一个iteration包括了一个step中前向传播、损失计算、反向传播和参数更新的流程。
            当然,在某些情况下,step和iteration可能会有细微的区别
            ——iteration是指完成一次前向传播和反向传播的过程,
            而step是指通过优化算法对模型参数进行一次更新的操作,当micro batch>1时两者不一样。
            """

    end_time = time.time()
    # avg_iteration_time = (end_time - start_time) / (num_epochs * len(train_dataloader))
    # avg_tput = (batch_size * input_size * len(train_dataloader) * epoch) / (
    #     end_time - start_time
    # )
    # if accelerator.device == "cuda:0":
    # if accelerator.is_main_process:
    #     wandb.log(
    #         {
    #             "avg_iteration_time": avg_iteration_time,
    #             "avg_tput per gpu (bs*seqlen/iter_time)": avg_tput,
    #             "avg_tput global (bs*seqlen/iter_time)": 8 * avg_tput,
    #         },
    #         step=0,
    #     )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="deepspeed baseline")
    parser.add_argument("--seq_len", type=int, default=1024, help="input seq len.")
    parser.add_argument("--batch", type=int, default=1, help="step batch size.")
    parser.add_argument("--gpu", type=int, default=8, help="gpu number.")
    parser.add_argument("--max_iter", type=int, default=10, help="每次实验最多运行多少次iteration")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument("--use_better_transformer", action="store_true", default=True)
    parser.add_argument("--use_flash_attn", action="store_true", default=True)
    # parser.add_argument(
    #     "--name", required=False, type=str, help="Name of the experiment for wandb"
    # )  # 需要指定
    parser.add_argument(
        "--wb_project_prefix",
        default="deepspeed baseline without cpu offload (Jan4,'24)",
        type=str,
        help="Project prefix in wandb",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Radom seed. If positive seed will be set as rando seed while negative one will be ignored.",
    )
    parser.add_argument("--model_name_or_path", type=str)
    args = parser.parse_args()

    import wandb

    if args.seed < 0:
        args.seed = int(time.time())

    main(args)
