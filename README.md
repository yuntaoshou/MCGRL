[stars-img]: https://img.shields.io/github/stars/yuntaoshou/MCGRL?color=yellow
[stars-url]: https://github.com/yuntaoshou/MCGRL/stargazers
[fork-img]: https://img.shields.io/github/forks/yuntaoshou/MCGRL?color=lightblue&label=fork
[fork-url]: https://github.com/yuntaoshou/MCGRL/network/members
[AKGR-url]: https://github.com/yuntaoshou/MCGRL

# Masked contrastive graph representation learning for age estimation
By Yuntao Shou, Xiangyong Cao, Huan Liu, Deyu Meng. [[arXiv link]](https://arxiv.org/abs/2306.17798)

[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]

This is an official implementation of 'Masked contrastive graph representation learning for age estimation' :fire:. Any problems, please contact shouyuntao@stu.xjtu.edu.cn. Any other interesting papers or codes are welcome. If you find this repository useful to your research or work, it is really appreciated to star this repository :heart:.

<div  align="center"> 
  <img src="https://github.com/yuntaoshou/MCGRL/blob/main/1.png" width=100% />
</div>



## 🚀 Installation

```bash
Pytorch 1.7.0,
timm 0.3.2,
torchprofile 0.0.4,
apex
```

## Training
Morph dataset
```bash
python train_MORPH.py --dataset MORPH --lr 0.003 --l2 0.0003 --dropout 0.5 --epochs 100 --w_loss1 1 --w_loss2 1 --w_loss3 1 --margin1 0.8 --margin2 0.2 --NN 4
```

FGNET dataset
```bash
python train_MORPH.py --dataset FGNET --lr 0.0005 --l2 0.00003 --dropout 0.5 --epochs 100 --w_loss1 1 --w_loss2 1 --w_loss3 1 --margin1 0.8 --margin2 0.2 --NN 4
```


CACD dataset
```bash
python train_MORPH.py --dataset CACD --lr 0.001 --l2 0.00003 --dropout 0.5 --epochs 120 --w_loss1 1 --w_loss2 1 --w_loss3 1 --margin1 0.8 --margin2 0.2 --NN 4
```

If our work is helpful to you, please cite:
```bash
@article{SHOU2024110974,
title = {Masked contrastive graph representation learning for age estimation},
journal = {Pattern Recognition},
pages = {110974},
year = {2024},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2024.110974},
url = {https://www.sciencedirect.com/science/article/pii/S0031320324007258},
author = {Yuntao Shou and Xiangyong Cao and Huan Liu and Deyu Meng},
}
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuntaoshou/MCGRL&type=Date)](https://star-history.com/#yuntaoshou/MCGRL&Date)
