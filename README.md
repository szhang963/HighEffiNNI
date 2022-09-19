# HighEffiNNI

This repository is a high-efficiency training framework with [NNI](https://github.com/microsoft/nni) for PyTorch. 

## Requirements

To install requirements:

```setup
PyTorch >= 1.6
NNI >= 2.9
easydict >= 1.9
```
## News
- In NNI 2.9, we can directly launch the visual website of NNI instead of configuring an ssh tunnel.
- Now, we can integrate the NNI's hyperparameter tuning into your project in a simple and quick way. You can compare [nni_minist.py](https://github.com/szhang963/HighEffiNNI/blob/main/nni_minist.py) and [minist.py](https://github.com/szhang963/HighEffiNNI/blob/main/minist.py) to find these.
- A solution about the error (Failed to establish a new connection) in 2.8 or 2.9 of NNI can be found in `[config.yml](https://github.com/szhang963/HighEffiNNI/blob/main/config.yml#L8)` 

## config.yml
A example of `config.yml` can be found in [here](https://github.com/szhang963/HighEffiNNI/blob/main/config.yml).
More experiment config references can see [here](https://nni.readthedocs.io/en/stable/reference/experiment_config.html).

## Quick start

To start up the NNI, run this command:

```
nnictl create --config config.yml -p 8140
```

To watch the running of NNI, run this command:
```
nnictl top 
```

To stop the NNI, run this command:
```
nnictl stop [--all] ([id])
```

TODO:
- [ ] To integrate the distributed training framework horovod into NNI.
- [ ] To develop more efficient training skills.
