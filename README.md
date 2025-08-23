# 📡 OFDM-NOMA System Simulation with Deep Learning Detection  

![MATLAB](https://img.shields.io/badge/MATLAB-R2020b%2B-orange?logo=mathworks)  
![Deep Learning Toolbox](https://img.shields.io/badge/Toolbox-Deep%20Learning-blue)  
![Signal Processing Toolbox](https://img.shields.io/badge/Toolbox-Signal%20Processing-green)  
![License](https://img.shields.io/badge/License-MIT-brightgreen)  
![Stars](https://img.shields.io/github/stars/DinhThien-pt/Deep-Learning-for-Signal-Detection-in-NonOrthogonal-Multiple-Access-Wireless-Systems?style=social)  
![Forks](https://img.shields.io/github/forks/DinhThien-pt/Deep-Learning-for-Signal-Detection-in-NonOrthogonal-Multiple-Access-Wireless-Systems?style=social)  

This repository contains MATLAB code for simulating a **Non-Orthogonal Multiple Access (NOMA)** system integrated with **Orthogonal Frequency Division Multiplexing (OFDM)** for two users. The system models signal transmission over multipath channels, performs channel estimation, power allocation, and symbol detection using traditional methods (**Successive Interference Cancellation - SIC, Maximum Likelihood - ML**) and a **Deep Neural Network (DNN) with LSTM layers**.  

Evaluations compare **Symbol Error Rate (SER)** under varying conditions such as SNR, Cyclic Prefix (CP) length, number of pilots, batch sizes, and learning rates.  

---

## 🔎 System Overview  

### ⚙️ Channel and Power Allocation  
- Generates a **static multipath channel** (20 paths for User 1, 12 for User 2).  
- Allocates power to achieve target SNRs (e.g., 12 dB), assigning **higher power to the weaker user** per subcarrier.  

### 📡 Data Transmission and Reception  
- Uses **comb-type pilots** for channel estimation (variable spacing).  
- Superimposes **QPSK data symbols** with allocated powers.  
- Performs **OFDM modulation**: IFFT, CP insertion, channel convolution, and AWGN noise addition.  
- Receiver: Removes CP, applies FFT.  

### 🧮 Channel Estimation  
- Implements **Least Squares (LS)** and **Minimum Mean Square Error (MMSE)** estimation.  
- Uses **spline interpolation** for non-pilot subcarriers.  

### 🔑 Symbol Detection  
- **Traditional Methods**:  
  - **SIC**: Decodes strong user first (zero-forcing), subtracts interference, then decodes weak user.  
  - **ML**: Exhaustive search over all QPSK symbol combinations assuming perfect CSI.  
- **Deep Learning Method**:  
  - **DNN (LSTM-based)** classifies superimposed symbols.  
  - Input: 384-dimensional vectors (real & imaginary parts of 64 subcarriers × 3 symbols).  
  - Output: **16 classes** (QPSK combinations for two users).  

### 🏋️ Training Data Generation  
- Fixed random seed for reproducibility.  
- Packets with **fixed pilots & varying QPSK symbols**.  
- Transmission at high SNR (40 dB) for noisy training samples.  
- Labels: **16 classes for QPSK pairs**.  

### 🤖 Neural Network Training  
- **Architecture**:  
  `Sequence Input → Flatten → LSTM (128 units) → Fully Connected (16 classes) → Softmax → Classification`  
- Optimizer: **Adam** (initial LR=0.01, drop factor=0.1).  
- Default: 20,000 mini-batch, 100 epochs.  
- Variations: Different learning rates, batch sizes, pilots, and CP lengths.  

### 📊 Evaluation  
- Tests across **SNRs (0–26 dB)**.  
- Computes **SER per user**.  
- Compares **DL, SIC-LS, SIC-MMSE, ML (perfect CSI)**.  
- Generates **SER vs. SNR plots**.  

---

## 📐 System Parameters  

| Parameter | Value |
|-----------|-------|
| Subcarriers | 64 |
| Users | 2 |
| Symbols per Packet | 3 (2 pilots + 1 data) |
| Modulation | QPSK (normalized to 1/√2) |
| Cyclic Prefix | 12 or 20 |
| Pilot Subcarriers | 16 or 64 |
| Training Samples | 80,000 (16 classes × 5000 packets/class) |
| Testing Packets | 500 per SNR point |

---

## 📂 File Structure  

### 🔧 Core Functions  
- `allocatePower.m` → Power allocation.  
- `channelEstimation.m` → LS & MMSE estimation.  
- `dataTransmissionReception.m` → Simulates OFDM-NOMA transmission/reception.  
- `detectML.m` → ML detection.  
- `getFeatureAndLabel.m` → Feature vector & label construction.  
- `symbolDecodeDL.m` → Deep learning detection.  
- `symbolDecodeSIC.m` → SIC detection.  

### 📑 Training Data Scripts  
- `trainDataCP12.m` → CP=12, 64 pilots.  
- `trainDataCP20.m` → CP=20, 64 pilots.  
- `trainDataCP12_16pilots.m` → CP=12, 16 pilots.  
- `trainDataCP12_64pilots.m` → CP=12, 64 pilots.  

### 🧠 Neural Network Training Scripts  
- `trainNNCP12.m` → Trains NN for CP=12, 64 pilots.  
- `trainNNCP20.m` → Trains NN for CP=20, 64 pilots.  
- `trainNNCP12_16pilots_lr*.m` → Trains with different learning rates.  
- `trainNNCP12_64pilots_bs*.m` → Trains with different batch sizes.  

### 📈 Evaluation Scripts  
- `Eval_CP_U1_U2.m` → SER for CP=12 vs. CP=20.  
- `Eval_BatchSize_U1_U2.m` → SER for batch sizes.  
- `Eval_LearnRate_U1_U2.m` → SER for different learning rates.  
- `Eval_Pilot_U1_U2.m` → SER for 16 vs. 64 pilots.  

---

## 🚀 Execution Order 
## To run the full system, execute the scripts in this order:
1. **Generate Training Data**  
   ```matlab
   trainDataCP12.m → Saves trainDataCP12.mat
   trainDataCP20.m → Saves trainDataCP20.mat
   trainDataCP12_16pilots.m → Saves trainDataCP12_16pilots.mat
   trainDataCP12_64pilots.m → Saves trainDataCP12_64pilots.mat
   Note: Uses fixed random seed for reproducibility. Each script takes ~10–20 seconds.

2. **Train Neural Networks**
   ```matlab
   Baseline: trainNNCP12.m, trainNNCP20.m → Saves NNCP12.mat, NNCP20.mat
   Pilot ablation: trainNNCP12_16pilots.m, trainNNCP12_64pilots.m
   Learning rate ablation: trainNNCP12_16pilots_lr*.m (vary LR)
   Batch size ablation: trainNNCP12_64pilots_bs*.m (vary batch size)
   Note: Training takes ~1–5 minutes per script (hardware-dependent, requires Deep Learning Toolbox).
   
3. **Evaluate and Plot Results**
   ```matlab
   Eval_CP_U1_U2.m: Compares CP lengths, saves SER_User1_CP20_vs_CP12.png and SER_User2_CP20_vs_CP12.png
   Eval_Pilot_U1_U2.m: Compares pilot counts, saves SER_User1_16_vs_64_pilots.png and SER_User2_16_vs_64_pilots.png
   Eval_LearnRate_U1_U2.m: Compares learning rates, saves SER_vs_SNR_DNN_LearningRates.png
   Eval_BatchSize_U1_U2.m: Compares batch sizes, saves SER_vs_SNR_DNN_BatchSizes.png
   Note: Each evaluation takes ~1–2 minutes, loading pre-trained networks and data.

## 📦 Dependencies
    ```bash
    MATLAB: Tested on R2020b or later
    Deep Learning Toolbox: For LSTM and neural network training
    Signal Processing Toolbox: For FFT/IFFT operations.
    
## 📝 Notes
- All scripts clear variables and close figures for clean execution.
- Plots are saved as PNG files in the working directory.
- Customize experiments by modifying parameters like numPacket, SNRs, or target SNRs in scripts.
- Assumes perfect power allocation knowledge at the receiver.
- For extensions (e.g., more users or modulations), modify core functions accordingly.
- If errors occur, ensure all files are in the MATLAB path and run scripts in the specified order.


 
