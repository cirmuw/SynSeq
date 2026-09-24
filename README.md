# SynSeq

This is the official repository for the Paper [SynSeq: End-to-End SYNTAX Score Prediction from Coronary Angiography Videos](http://arxiv.org/abs/2609.27696). 
You will find steps to reproduce the paper here. If you have any issues open an issue here on GitHub or reach us otherwise, we will try to help as good as we can.


# Preparation

Download the CardioSyntax dataset from [the official repository](https://zenodo.org/records/14005818).

The data split is located inside the "dataset" folder. It includes two folders for pretraining and final training. The files starting with "step2" are our split containing only the 60 patients with measurements of three experts. "validation" describes the extended split with more testdata and less training data (note that for the extended split not for all test data labels an annotation of all 3 experts is visible).

## Install

Install the packages listed in requirements.txt. Special measures might have to be taken for cuda installation.

# Execution

We give a brief overview on how to run the code, it has been tested locally on a CPU and on a scientific cluster.

## Configuration

The default config is provided, it can be adjusted as needed (e.g. for data paths). 
Please read the  [Hydra documentation](https://hydra.cc/docs/intro/) on how to swap out configuration parts.

The "8_by_8" configuration in conjunction with the "fast_dev_run" option can be used to quickly verify if the environment is working and the code can be run.
## Training

Run the backbone for each side:
```bash
python backbone_train.py dataset.root=/path/to/dataset/ dataset.artery=left
python backbone_train.py dataset.root=/path/to/dataset/ dataset.artery=right
```

Run full model training for each side:
```bash
python seq_train.py dataset.root=/path/to/dataset/ dataset.artery=left
python seq_train.py dataset.root=/path/to/dataset/ dataset.artery=right
```
## Inference

Find the correct output weights and run inference:

```bash
python seq_apply.py dataset.root=/path/to/dataset/ model.model_path_left=outputs/YYYY-MM-DD/hh-mm-ss/result_post.pt model.model_path_right=outputs/YYYY-MM-DD/hh-mm-ss/result_post.pt

```
Note that the left and right model weights are stored in different folders (and can be distinguished/automatically found by the config stored in the .hydra folder for each run).


# Results

The results are stored in the corresponding "outputs" folder. It contains the full dataset, the newly generated results are added in the "predictions" object for each patient.
Note that zero-clamping is not automatically included and must be performed during analysis:
```python
    df["left_syntax"] = df["left_syntax"].clip(lower=0)
    df["right_syntax"] = df["right_syntax"].clip(lower=0)
    df["syntax"] = df["left_syntax"] + df["right_syntax"]
```

Also, if reproducing the "log-space" ablation, the outputs must be mapped correctly:
```python
    df["left_syntax"] = np.exp(df["left_syntax"] / 70) - 1
    df["right_syntax"] = np.exp(df["right_syntax"] / 70) - 1

    df["syntax"] = df["left_syntax"] + df["right_syntax"]
```
