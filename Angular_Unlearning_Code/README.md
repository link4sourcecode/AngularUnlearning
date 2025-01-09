# Angular Unlearning


## Reproduction Guide
### Getting Started 
To run this repository, we kindly advise you to install Python 3.8 and PyTorch 1.12.0 with Anaconda. You can download Anaconda and read the installation instruction on the official website\footnote{https://www.anaconda.com/download/}.

Create a new virtual environment named ``env1'' based on Python 3.8 and enter this environment:
```
conda create -n env1 python=3.8.16
conda activate env1
```
Install PyTorch and related packages in torch environment:
```
conda install pytorch==1.12.0+cu101
conda install torchvision -c pytorch
pip install numpy==1.23.5 scikit-learn==1.0.2 opencv-python==4.7.0 pillow==9.4.0
```

 This guide outlines the structure of the Python files and their corresponding functions in our project, facilitating the reproduction of our results.

The model architectures are defined in the following file: 
```
./Angular_Unlearning_Code/models.py 
```

The core training functions are implemented in: 
```
./Angular_Unlearning_Code/trainer.py 
```

The main procedure of our Angular Unlearning approach is detailed in the following Jupyter notebook: 
```
./Angular_Unlearning_Code/angular_unlearning.ipynb 
```

The key implementation of Angular Unlearning is in the code cell labeled **angular unlearning**. The primary evaluation metrics are provided in the **membership\_attack()** function located in the **MIA** code cell, and the **eval()** function within **trainer.py**