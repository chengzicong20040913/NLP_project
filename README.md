# NLP_project
**Author: Zicong Cheng**
这是一份代码使用说明
## 格式说明
在项目目录下你需要看到如下构成
```bash
.
├── alpaca-cleaned
├── evals
├── Finetune_autoregressive_all.py
├── Finetune_autoregressive_output.py
├── Finetuned-Qwen2.5-0.5B
├── finetune-eval.ipynb
├── Finetune_origin.py
├── logs
├── outputs
├── Qwen2.5-0.5B
├── test_output.py
└── tmp
```
其中每个文件的解释如下
- alpaca-cleaned为所用的数据集
- Qwen2.5-0.5B存放的是原始的模型
- Finetuned-Qwen2.5-0.5B存放的是微调后的模型
- evals存放的是评估结果
- logs存放的是训练日志
- outputs存放的是输出结果
- tmp存放的临时文件
- Finetune_origin.py为原始模型的微调代码
- Finetune_autoregressive_all.py为自回归模型的微调代码
- Finetune_autoregressive_output.py为自回归模型的只计算输出损失的微调代码
- test_output.py为测试输出代码
- finetune-eval.ipynb为填充后的原始文件

在evals文件夹下，你会看到如下文件夹
```bash
.
├── plm
└── sft
```
plm文件夹下存放的是原始模型的评估结果，sft文件夹下存放的是自回归模型的评估结果.

## 使用说明
### 微调
- 原始的模型微调直接运行Finetune_origin.py即可，自回归模型微调运行Finetune_autoregressive_all.py即可，自回归模型只计算输出损失的微调运行Finetune_autoregressive_output.py即可，运行完毕后会在outputs文件夹下看到最后一轮过后的模型文件夹打开后看到`model.safetensors`,将其复制到Finetuned-Qwen2.5-0.5B文件夹下替换原来的`model.safetensors`即可。
- 评测使用finetune-eval.ipynb,运行最后三个cell即可，分别是模型路径设置，原始模型评估，微调模型评估。

### 测试输出
- 测试输出使用test_output.py,运行后会直接输出结果，可以根据需要修改代码中的输入文本，输出文本，模型路径等参数。这里要注意命令行需要你的模型路径，输入文本可自行在代码中修改不做参数传递。
