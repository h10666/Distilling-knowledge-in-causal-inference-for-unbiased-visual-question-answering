# CIKD

## 论文简介

### 动机

![motivation](./assets/motivation.jpg)

Distilling knowledge in causal inference for unbiased visual question answering

现有的视觉问答模型倾向于学习问题与答案之间的伪相关，忽视了视觉信息对答案的作用。当训练集分布与测试集分布不一致时，这些视觉问答模型的性能会显著下降。为解决这一问题，本文提出了一种新的无偏视觉问答模型CKCL。CIKD使用基于因果推断的知识蒸馏来降低问题与答案之间的伪相关，并使用课程自步学习来挖掘对模型有益的好偏差。本文首先构建一个因果图来分析反事实的因果关系，并根据因果效应来获取因果目标。然后，本文使用知识蒸馏将因果目标中的知识迁移到传统的视觉问答模型中。此外，为解决由于知识蒸馏引入的偏差，本文构建集成学习模型来降低引入的偏差对模型的影响。并且，通过实验结果验证了CIKD的有效性。对比现有的方法，CIKD的性能在VQA-CP v2数据集上的显著提升很好地验证了本文工作的贡献。

### 模型结构

![arch](./assets/CIKD_arch.jpg)

## 数据集下载

使用命令下载和解压预提取的视觉特征，命令如下
```
wget http://data.lip6.fr/cadene/block/coco/extract_rcnn/2018-04-27_bottom-up-attention_fixed_36.tar
tar -xvf 2018-04-27_bottom-up-attention_fixed_36.tar
```
将解压后的文件存放到data/trainval_features文件夹下，如没有该文件夹可以自行创建

## 测试

测试模型

```
python evaluate_ci.py
```

其中，可以在evaluate_ci.py更换权重，VQA-CP v数据集的权重均存在./save_models/exp0里

./saved_models/exp0/lhm_model_epoch14_lm_kd_ci_abs_weight.pth 是基于abs的权重
./saved_models/exp0/lhm_model_epoch14_lm_kd_ci_sub_weight.pth 是给予sub的权重

