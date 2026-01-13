---
tasks:
- sentence-embedding

model-type:
- bert

domain:
- nlp

frameworks:
- pytorch

backbone:
- transformer

metrics:
- mrr@10
- recall@1000
- ndcg@10

license: Apache License 2.0

language:
- en

tags:
- text representation
- text retrieval
- passage retrieval
- Transformer
- GTE
- 文本相关性
- 文本相似度

---

# GTE英文通用文本表示模型

文本表示是自然语言处理(NLP)领域的核心问题, 其在很多NLP、信息检索的下游任务中发挥着非常重要的作用。近几年, 随着深度学习的发展，尤其是预训练语言模型的出现极大的推动了文本表示技术的效果, 基于预训练语言模型的文本表示模型在学术研究数据、工业实际应用中都明显优于传统的基于统计模型或者浅层神经网络的文本表示模型。这里, 我们主要关注基于预训练语言模型的文本表示。

文本表示示例, 输入一个句子, 输入一个固定维度的连续向量:

- 输入: how long it take to get a master's degree
- 输出: [0.27162,-0.66159,0.33031,0.24121,0.46122,...]

文本的向量表示通常可以用于文本聚类、文本相似度计算、文本向量召回等下游任务中。

## 文本表示模型

基于监督数据训练的文本表示模型通常采用Dual Encoder框架, 如下图所示。在Dual Encoder框架中, Query和Document文本通过预训练语言模型编码后, 通常采用预训练语言模型[CLS]位置的向量作为最终的文本向量表示。基于标注数据的标签, 通过计算query-document之间的cosine距离度量两者之间的相关性。

<div align=center><img width="450" height="300" src="./resources/dual-encoder.png" /></div>


模型训练方法请参考论文[Towards General Text Embeddings with Multi-stage Contrastive Learning](https://arxiv.org/abs/2308.03281)。

## 使用方式和范围

使用方式:
- 直接推理, 对给定文本计算其对应的文本向量表示，向量维度384

使用范围:
- 本模型可以使用在通用领域的文本向量表示及其下游应用场景, 包括双句文本相似度计算、query&多doc候选的相似度排序

### 如何使用

在ModelScope框架上，提供输入文本(默认最长文本长度为128)，即可以通过简单的Pipeline调用来使用coROM文本向量表示模型。ModelScope封装了统一的接口对外提供单句向量表示、双句文本相似度、多候选相似度计算功能

#### 代码示例
```python
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = "damo/nlp_gte_sentence-embedding_english-small"
pipeline_se = pipeline(Tasks.sentence_embedding,
                       model=model_id)

# 当输入包含“soure_sentence”与“sentences_to_compare”时，会输出source_sentence中首个句子与sentences_to_compare中每个句子的向量表示，以及source_sentence中首个句子与sentences_to_compare中每个句子的相似度。
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

model_id = "damo/nlp_gte_sentence-embedding_english-small"
pipeline_se = pipeline(Tasks.sentence_embedding, 
                       model=model_id)

inputs = {
        "source_sentence": ["how long it take to get a master degree"],
        "sentences_to_compare": [
            "On average, students take about 18 to 24 months to complete a master degree.",
            "On the other hand, some students prefer to go at a slower pace and choose to take",
            "several years to complete their studies.",
            "It can take anywhere from two semesters"
        ]
    }


result = pipeline_se(input=inputs)
print (result)
'''
# 当输入仅含有soure_sentence时，会输出source_sentence中每个句子的向量表示以及首个句子与其他句子的相似度。
inputs = {
        "source_sentence": [
            "On average, students take about 18 to 24 months to complete a master degree.",
            "On the other hand, some students prefer to go at a slower pace and choose to take",
            "several years to complete their studies.",
            "It can take anywhere from two semesters"
        ]
    }
result = pipeline_se(input=inputs2)
print (result)
```

**默认向量维度384, scores中的score计算两个向量之间的内积距离得到**

### 模型局限性以及可能的偏差

本模型基于英文领域通用数据集训练，在垂类领域英文文本上的文本效果会有降低，请用户自行评测后决定如何使用


### 训练示例代码

```python
# 需在GPU环境运行
# 加载数据集过程可能由于网络原因失败，请尝试重新运行代码
from modelscope.metainfo import Trainers                                                                                                                                                              
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
import tempfile
import os

tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

# load dataset
ds = MsDataset.load('msmarco-passage-ranking', 'zyznull')
train_ds = ds['train'].to_hf_dataset()
dev_ds = ds['dev'].to_hf_dataset()
model_id = 'damo/nlp_gte_sentence-embedding_english-small'
def cfg_modify_fn(cfg):
    cfg.task = 'sentence-embedding'
    cfg['preprocessor'] = {'type': 'sentence-embedding','max_length': 256}
    cfg['dataset'] = {
        'train': {
            'type': 'bert',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'text_fileds': ['text'],
            'qid_field': 'query_id'
        },
        'val': {
            'type': 'bert',
            'query_sequence': 'query',
            'pos_sequence': 'positive_passages',
            'neg_sequence': 'negative_passages',
            'text_fileds': ['text'],
            'qid_field': 'query_id'
        },
    }
    cfg['train']['neg_samples'] = 4
    cfg['evaluation']['dataloader']['batch_size_per_gpu'] = 30
    cfg.train.max_epochs = 1
    cfg.train.train_batch_size = 4
    return cfg 
kwargs = dict(
    model=model_id,
    train_dataset=train_ds,
    work_dir=tmp_dir,
    eval_dataset=dev_ds,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(name=Trainers.nlp_sentence_embedding_trainer, default_args=kwargs)
trainer.train()
```

### 模型效果评估

英文多任务向量评测榜单MTEB结果如下：

| Model Name | Model Size (GB) | Dimension | Sequence Length | Average (56) | Clustering (11) | Pair Classification (3) | Reranking (4) | Retrieval (15) | STS (10) | Summarization (1) | Classification (12) |
|:----:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| [**gte-large**](https://huggingface.co/thenlper/gte-large) | 0.67 | 1024 | 512 | **63.13** | 46.84 | 85.00 | 59.13 | 52.22 | 83.35 | 31.66 | 73.33 |
| [**gte-base**](https://huggingface.co/thenlper/gte-base) 	| 0.22 | 768 | 512 | **62.39** | 46.2 | 84.57 | 58.61 | 51.14 | 82.3 | 31.17 | 73.01 |
| [e5-large-v2](https://huggingface.co/intfloat/e5-large-v2) | 1.34 | 1024| 512 | 62.25 | 44.49 | 86.03 | 56.61 | 50.56 | 82.05 | 30.19 | 75.24 |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.44 | 768 | 512 | 61.5 | 43.80 | 85.73 | 55.91 | 50.29 | 81.05 | 30.28 | 73.84 |
| [**gte-small**](https://huggingface.co/thenlper/gte-small) | 0.07 | 384 | 512 | **61.36** | 44.89 | 83.54 | 57.7 | 49.46 | 82.07 | 30.42 | 72.31 |
| [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) | - | 1536 | 8192 | 60.99 | 45.9 | 84.89 | 56.32 | 49.25 | 80.97 | 30.8 | 70.93 |
| [e5-small-v2](https://huggingface.co/intfloat/e5-base-v2) | 0.13 | 384 | 512 | 59.93 | 39.92 | 84.67 | 54.32 | 49.04 | 80.39 | 31.16 | 72.94 |
| [sentence-t5-xxl](https://huggingface.co/sentence-transformers/sentence-t5-xxl) | 9.73 | 768 | 512 | 59.51 | 43.72 | 85.06 | 56.42 | 42.24 | 82.63 | 30.08 | 73.42 |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) 	| 0.44 | 768 | 514 	| 57.78 | 43.69 | 83.04 | 59.36 | 43.81 | 80.28 | 27.49 | 65.07 |
| [sgpt-bloom-7b1-msmarco](https://huggingface.co/bigscience/sgpt-bloom-7b1-msmarco) 	| 28.27 | 4096 | 2048 | 57.59 | 38.93 | 81.9 | 55.65 | 48.22 | 77.74 | 33.6 | 66.19 |
| [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) 	| 0.13 | 384 | 512 	| 56.53 | 41.81 | 82.41 | 58.44 | 42.69 | 79.8 | 27.9 | 63.21 |
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 	| 0.09 | 384 | 512 	| 56.26 | 42.35 | 82.37 | 58.04 | 41.95 | 78.9 | 30.81 | 63.05 |
| [contriever-base-msmarco](https://huggingface.co/nthakur/contriever-base-msmarco) 	| 0.44 | 768 | 512 	| 56.00 | 41.1 	| 82.54 | 53.14 | 41.88 | 76.51 | 30.36 | 66.68 |
| [sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base) 	| 0.22 | 768 | 512 	| 55.27 | 40.21 | 85.18 | 53.09 | 33.63 | 81.14 | 31.39 | 69.81 |


## 引用

```BibTeX
@misc{li2023general,
      title={Towards General Text Embeddings with Multi-stage Contrastive Learning}, 
      author={Zehan Li and Xin Zhang and Yanzhao Zhang and Dingkun Long and Pengjun Xie and Meishan Zhang},
      year={2023},
      eprint={2308.03281},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```