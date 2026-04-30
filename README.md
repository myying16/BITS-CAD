# BITS-CAD: Bidirectional Text-sketch Sequence Fusion for CAD Generation

## Installation
- Python 3.9
- Cuda 11.8+

Install python package dependencies through pip:

```bash
pip install -r requirements.txt
# Mamba install
pip install causal-conv1d==1.1.1
pip install mamba-ssm==1.1.1
```
## Dataset

please refer to the code from [here](https://github.com/lllssc/Drawing2CAD) and extract them under `data` folder.


## Training

To train the model in different input options:

```bash
python train.py --exp_name your_exp_name
```

Since different input options lead to different models, it is recommended to specify the experiment name using `--exp_name` for each run. For more configurable parameters and options, please refer to `config/config.py`.

## Test

After training the model, run the model to inference all test data:

```python
python test.py  --exp_name your_exp_name
```

## Evaluation
After inference, the final results will be saved under `proj/your_exp_name/test_results`. 

 To evaluate the results:

  ```bash
  $ cd evaluation
  # for command accuray and parameter accuracy
  $ python evaluate_ae_acc.py --src ../proj_log/your_exp_name/test_results
  # for chamfer distance and invalid ratio
  $ python evaluate_ae_cd.py --src ../proj_log/your_exp_name/test_results --parallel
  ```

To export and visualize the final CAD models, please refer to the code from [DeepCAD](https://github.com/ChrisWu1997/DeepCAD).

## Acknowledgement

This repository builds upon the following awesome datasets and projects:

- [DeepCAD](https://github.com/ChrisWu1997/DeepCAD)
- [Drawing2CAD](https://github.com/lllssc/Drawing2CAD)
- [Mamba](https://github.com/state-spaces/mamba)
- [Vision Mamba](https://github.com/hustvl/Vim)

