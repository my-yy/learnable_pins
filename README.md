# PyTorch implementation of Learnable Pins

> Learnable PINs: Cross-Modal Embeddings for Person Identity


This code is based on the [Self-Lifting](https://github.com/my-yy/sl_icmr2022) project which enables train a model just in minutes.

Note that the dataset splitting is VGG-Vox style, which is different from the original paper,
but you can still know how the Curriculum-based Mining is implemented 😜 (`utils/pair_selection_util.py`).


## Dataset

The dataset is the same as the Self-Lifting project. If you already have it, you can just create a soft link in the project root: 

`ln -s Your-Self-Lifting-Project-Root/dataset ./dataset`

Or you need to download it by referring to [Self-Lifting](https://github.com/my-yy/sl_icmr2022).


## Training

Just Run: ``python 1_pins.py``

You also can use [wandb](https://wandb.ai) to view the training process:

1. Create  `wb_config.json`  file in the  `./configs` folder, using the following content:

   ```
   {
     "WB_KEY": "Your wandb auth key"
   }
   ```
2. add `--dryrun=False` to the training command, for example:   `python 1_pins.py --dryrun=False`

## Results
Because the Backbone structure and test script are different from the original paper, the scores behave much higher.
![](results.png)


## Paper Explanation (Chinese Language) 
[【音脸关系学习】：Learnable Pins 论文解读与代码复现](https://zhuanlan.zhihu.com/p/557632629)


## Other Resources
[Voice Face Association Learning Papers & Codes](https://github.com/my-yy/vfal_papers) 
