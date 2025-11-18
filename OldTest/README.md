# 面向课堂场景的大模型定制化微调：逻辑问答交互与授课效果的关联性研究

## 项目背景和简介
在传统课堂话语分析体系中，使用特征工程的方法虽然效果不错，但需要耗费大量的人力物力；另一种使用 Transformer 等经典深度学习的方法虽然非常高效，但是典型的“黑盒”模型，可解释性弱，而在实际课堂分析中，教师往往看重可解释性。因此，本项目实现了：
- 通过使用微调大模型，在保证性能的同时，兼顾可解释性
- 编写相应的封装代码，实现端到端的程序调用

## 技术方案
本项目基于CDTB数据集，采用MindSpore框架微调DeepSeek-R1-Distill-Qwen-1.5B大模型，实现了中文篇章级句间关系识别任务。

## 技术栈与核心组件
| 技术栈           | 版本/说明                         |
| ---------------- | --------------------------------- |
| 深度学习框架     | MindSpore                         |
| NLP工具库        | mindnlp                           |
| 大语言模型       | DeepSeek-R1-Distill-Qwen-1.5B     |
| 参数高效微调方法 | LoRA (Low-Rank Adaptation)        |
| 数据集           | CDTB (Chinese Discourse TreeBank) |

## 核心技术实现细节

### 数据处理流程
```python
# 数据加载
df_train = pd.read_json(train_path)
ds_train = Dataset.from_pandas(df_train)

# 对话模板构建
def process_func(example):
    instruction = tokenizer(
        f"<|im_start|>system\n你是PDTB文本关系分析助手<|im_end|>\n"
        f"<|im_start|>user\n{example['content']}<|im_end|>\n"
        f"<|im_start|>assistant\n",
        add_special_tokens=False
    )
    response = tokenizer(f"{example['summary']}", add_special_tokens=False)

    # 输入序列构建与截断
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    input_ids = input_ids[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# 数据编码
tokenized_train = ds_train.map(process_func, remove_columns=ds_train.column_names)
```

### LoRA微调配置
```python
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=8,                    # LoRA秩
    lora_alpha=32,          # LoRA缩放因子
    lora_dropout=0.1,       # dropout率
    inference_mode=False    # 训练模式
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 输出可训练参数比例
```

### 模型训练与合并
```python
# 训练配置
args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=5,
    num_train_epochs=3,
    learning_rate=3e-5,
    save_steps=100
)

# 训练执行
trainer = Trainer(model=model, args=args, ...)
trainer.train()

# LoRA权重合并
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()
model.save_pretrained(merged_path)
```


## 数据集准备
### 原始数据
本项目使用哈尔滨工业大学 CDTB 数据集，并在此基础上进行了一定的补充，保障了数据的规模、多样性和高质量。各位开发者可通过`https://ir.hit.edu.cn/2024/1029/c19699a357757/page.htm`向哈工大官方网站申请调用。

### 数据集介绍
哈工大 CDTB 即 HIT - CDTB（哈尔滨工业大学中文篇章关系语料），核心借鉴了 PDTB 的标注标准与研究框架，将篇章级中文语义共分为：
- 扩展
- 因果
- 比较
- 并列
- 时序
- 条件

其中，由于**时序**和**条件**两类的样本数量过少，本实验中将其统一为**其他**关系

### 数据集处理
为保证实验效果，我们在 CDTB 数据集的基础上，做出了一定的修改：
1. 原始样本格式为：
    ```
    {
        "content": "星汉是什么？银河。",
        "summary": "扩展"
    }
    ```
    为解决可解释性问题，我们决定扩充数据集，为数据加上具体的原因类标签而非只是分类结果。扩充后的样本格式为：
    ```
    {
        "content": "他的有没有什么不足之处？我觉得他可以就是加一些他自己的感受，因为他如果光只说那些一系列的动作，就感觉很空白，没有什么情感在里面。",
        "summary": "扩展\n原因：前半句话提出问题，询问他的不足之处，后半句话则具体回答了我认为的他的不足之处，所以属于扩展关系。"
    }
    ```
2. 由于时间有限，我们只从一万条数据中随机抽取了两千条数据，并为他们手动打上了原因。但为保持微调效果，我们为其余数据统一添加了“原因：”字样，即：
   ```
    {
        "content": "星汉是什么？银河。",
        "summary": "扩展\n原因："
    }
   ```
3. 训练集和测试集的比例按 8 : 2 的比例在保证分类比例不变的情况下随机划分


## 项目使用
本项目在华为云上实现
### 环境搭建
1. **安装MindSpore**
   ```bash
   pip install mindspore
   ```

2. **安装依赖库**
   ```bash
   pip install mindnlp datasets transformers peft
   ```

### 数据准备
1. 准备 CDTB 格式的 JSON 数据集文件
2. 数据集格式要求：
   ```json
   {
     "content": "文本内容",
     "summary": "关系类型\n原因：关系解释"
   }
   ```
3. 将数据集划分为train.json和val.json，放置于data目录下

### 模型训练
使用`train.ipynb`进行模型训练：
- 训练过程将生成output目录，保存checkpoint文件
- 可通过日志查看loss、epoch等训练指标

### LoRA权重合并
使用`merge.ipynb`合并LoRA权重与基础模型

## 模型性能与评估
- 训练集损失：0.9027（3个epoch）
- 可训练参数比例：0.5168%（仅微调900万参数）
- 训练时间：约1.6小时（3个epoch）