"""
The main program for finetuning LLMs with Huggingface Transformers Library.

ALL SECTIONS WHERE CODE POSSIBLY NEEDS TO BE FILLED IN ARE MARKED AS TODO.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import sys
import torch
from transformers import TrainingArguments, HfArgumentParser, Trainer, AutoTokenizer, AutoModelForCausalLM
import datasets
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# Define the arguments required for the main program.
# NOTE: You can customize any arguments you need to pass in.
@dataclass
class ModelArguments:
    """Arguments for model
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the LLM to fine-tune or its name on the Hugging Face Hub."
        }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype."
            ),
            "choices": ["bfloat16", "float16", "float32"],
        },
    )
    # TODO: add your model arguments here
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum length of the input sequence."
        }
    )
    
    


@dataclass
class DataArguments:
    """Arguments for data
    """
    dataset_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the fine-tuning dataset or its name on the Hugging Face Hub."
        }
    )
    # TODO: add your data arguments here
    #调整划分数据集的比例
    train_size: Optional[float] = field(
        default=0.8,
        metadata={
            "help": "The proportion of the training dataset."
        }
    )
@dataclass
class CustomTrainingArguments(TrainingArguments):
    train_batch_size:Optional[int] = field(
        default=2,
        metadata={
            "help": "The custom training batch size."
        }
    )
    learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={
            "help": "The custom learning rate."
        }
    )
    eval_batch_size: Optional[int] = field(
        default=2,  # 默认设置为2，视需要调整
        metadata={
            "help": "The batch size used for evaluation."
        }
    )
    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "The maximum gradient norm."
        }
    )
    logging_steps: Optional[int] = field(
        default=500,
        metadata={
            "help": "Log every X updates steps."
        }
    )
    warmup_steps: Optional[int] = field(
        default=500,
        metadata={
            "help": "The number of warmup steps."
        }
    )
    save_steps: Optional[int] = field(
        default=10000,
        metadata={
            "help": "Save checkpoint every X updates steps."
        }
    )
# The main function
# NOTE You can customize some logs to monitor your program.
def finetune():
    # TODO Step 1: Define an arguments parser and parse the arguments
    # NOTE Three parts: model arguments, data arguments, and training arguments
    # HINT: Refer to 
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/internal/trainer_utils#transformers.HfArgumentParser
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # TODO Step 2: Load tokenizer and model
    # HINT 1: Refer to
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/tokenizer#tokenizer
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/qwen2
    # HINT 2: To save training GPU memory, you need to set the model's parameter precision to half-precision (float16 or bfloat16).
    #         You may also check other strategies to save the memory!
    #   * https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/llama2#usage-tips
    #   * https://huggingface.co/docs/transformers/perf_train_gpu_one
    #   * https://www.53ai.com/news/qianyanjishu/2024052494875.html
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16 if model_args.torch_dtype is None else model_args.torch_dtype,
    )

    # TODO Step 3: Load dataset
    # HINT: https://huggingface.co/docs/datasets/v3.1.0/en/package_reference/main_classes#datasets.Dataset
    dataset = datasets.load_dataset(data_args.dataset_path)
    # HINT: You may need to adjust the dataset to fit the model's input format.

    # TODO Step 4: Define the data collator function
    # NOTE During training, for each model parameter update, we fetch a batch of data, perform a forward and backward pass,
    # and then update the model parameters. The role of the data collator is to process the data (e.g., padding the data within
    # a batch to the same length) and format the batch into the input required by the model.
    #
    # In this assignment, the purpose of the custom data_collator is to process each batch of data from the dataset loaded in
    # Step 3 into the format required by the model. This includes tasks such as tokenizing the data, converting each token into 
    # an ID sequence, applying padding, and preparing labels.
    # 
    # HINT:
    #   * Before implementation, you should:
    #      1. Clearly understand the format of each sample in the dataset loaded in Step 3.
    #      2. Understand the input format required by the model (https://huggingface.co/docs/transformers/model_doc/qwen2#transformers.Qwen2ForCausalLM).
    #         Reading its source code also helps!

    def data_collator(batch: List[Dict]):
        """
        batch: list of dict, each dict of the list is a sample in the dataset.
        """

        # 处理每个样本的 'instruction' 和 'input'，将它们拼接起来
        texts = [sample['instruction'] +'\n'+ sample['input'] for sample in batch]
        
        # 使用 tokenizer 对拼接后的文本进行 tokenization
        tokenized_inputs = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=model_args.max_length,
        )

        # 使用 tokenizer 对 'output' 进行 tokenization 作为标签
        tokenized_labels = tokenizer(
            [sample['output'] for sample in batch],  # 这里处理的是每个样本的 'output'
            truncation=True,
            padding="max_length",
            max_length=model_args.max_length,
        )

        # 对 causal language modeling，shift labels 向右
        input_ids = tokenized_inputs.input_ids
        attention_mask = tokenized_inputs.attention_mask
        labels = tokenized_labels.input_ids
        #要转换为torch格式
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # 返回处理后的 input_ids，attention_mask 和 labels
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        

    # TODO Step 5: Define the Trainer
    # HINT: https://huggingface.co/docs/transformers/main_classes/trainer
    
    train_set=dataset['train']
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_set,
        #callbacks=[LossLoggingCallback]  # 传入自定义回调
    )

    # Step 6: Train!
    trainer.train()
    
# Pass your training arguments.
# NOTE [IMPORTANT!!!] DO NOT FORGET TO PASS PROPER ARGUMENTS TO SAVE YOUR CHECKPOINTS!!!
#os.environ["TOKENIZERS_PARALLELISM"] = "false"#不要并行处理
sys.argv = [
    "notebook", 
    "--model_name_or_path", "Qwen2.5-0.5B",
    "--dataset_path", "alpaca-cleaned",
    "--output_dir", "outputs",
    "--max_length", "1024",
    "--train_size", "0.9",
    "--remove_unused_columns", "False",
    "--torch_dtype", "bfloat16",#这里不要float16，会报错
    "--per_device_eval_batch_size","2",
    "--train_batch_size","4",
    "--fp16","True",
    "--num_train_epochs", "1",  # 指定训练的轮数
]
finetune()
    