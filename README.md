# SPIRAL

### Introduction
This is a Python implementation of **SPIRAL**, a signal-power integrity co-analysis framework for high-speed inter-chiplet serial links validation. (https://ieeexplore.ieee.org/document/10473908)

The framework first builds equivalent models for the links with a **machine learning** based transmitter model and the **impulse response** of the channel and receiver. Then, the signal-power integrity is co-analyzed with a **pulse response** based method using the equivalent models. The framework calculates the output signals corresponding to the given input data and generates eye diagrams.

## Dependencies
- Python = 3.8
- PyTorch >= 1.2.0
- Python packages: 
  - numpy
  - scipy
  - skrf

## Usage
First, prepare the trained TX model and S-parameters of the channel-RX. A few examples are provided, please refer to `./checkpoint` to find TX models and `./link_rx` to find channel-RX models. 

Then, build the info file as `b1_l3.txt` and `b2_s3.txt`, which defines the models, UIs, S-parameters information, $Z_0$, $V_p$, dc resistance of channel, $C_L$, rising/falling time, input amplitude, input data, step and whether to consider the PSN and de-emphasis. 

Run `spiral_sipi.py` to obtain the output signals and eye diagrams.
