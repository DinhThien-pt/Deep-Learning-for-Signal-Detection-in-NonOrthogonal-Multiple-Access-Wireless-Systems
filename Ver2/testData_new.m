clear variables;
close all;

% Load training data and essential parameters
load('trainDataCP12_16pilots.mat','h','numPSC_16pilots','idx_sc','fixedPilot_16pilots');
load('trainDataCP12_64pilots.mat','numPSC_64pilots','fixedPilot_64pilots');
% Load neural networks
load('NNCP12_16pilots.mat','net'); % Network for 16 pilots
net_16pilots = net;
load('NNCP12_64pilots.mat','net'); % Network for 64 pilots
net_64pilots = net;

% System parameters
[numPath,numUE] = size(h);
numSC = 64; % number of subcarriers
numPSym = numUE; % number of pilot OFDM symbols per packet
numDSym = 1; % number of data OFDM symbol per packet
numSym = numPSym+numDSym; % number of OFDM symbols per packet
pilotSpacing_16 = numSC/numPSC_16pilots; % Pilot spacing for 16 pilots
pilotSpacing_64 = numSC/numPSC_64pilots; % Pilot spacing for 64 pilots
pilotStart = [1,1]; % pilot starting subcarrier for two users 
lengthCP = 12; % Fixed CP length as trained

% QPSK modulation
constQPSK = [1-1j;1+1j;-1+1j;-1-1j]; % QPSK constellation
% Labels
symClass = [constQPSK(1) constQPSK(1); constQPSK(1) constQPSK(2); constQPSK(1) constQPSK(3); constQPSK(1) constQPSK(4);
            constQPSK(2) constQPSK(1); constQPSK(2) constQPSK(2); constQPSK(2) constQPSK(3); constQPSK(2) constQPSK(4);
            constQPSK(3) constQPSK(1); constQPSK(3) constQPSK(2); constQPSK(3) constQPSK(3); constQPSK(3) constQPSK(4);
            constQPSK(4) constQPSK(1); constQPSK(4) constQPSK(2); constQPSK(4) constQPSK(3); constQPSK(4) constQPSK(4)];
labelClass = 1:1:size(symClass,1);
% Testing data size
numPacket = 500; % Reduce numPacket to increase noise effect
fixedPilot_16 = repmat(fixedPilot_16pilots,1,1,1,numPacket); % For 16 pilots
fixedPilot_64 = repmat(fixedPilot_64pilots,1,1,1,numPacket); % For 64 pilots

% Power allocations
targetSNR_1 = 12; % dB, target SNR for strong user
targetSNR_2 = 12; % dB, target SNR for weak user
targetSNR_linear_1 = 10^(targetSNR_1/10);
targetSNR_linear_2 = 10^(targetSNR_2/10);
H = fft(h,numSC,1); 
gainH = (abs(H).^2).';

% Noise computation
EsN0_dB = 0:2:28; % Start from 0 dB to match Fig. 5 range
EsN0 = 10.^(EsN0_dB./10);
symRate = 2; % symbol rate, 2 symbol/s
Es = 1; % symbol energy, joules/symbol
sigPower = Es*symRate; % total signal power, watts
symPower = sigPower/numUE; % signal power per symbol 
N0 = sigPower./EsN0; % noise power in watts/Hz
bw = 1; % bandwidth per subcarrier, Hz
nPower = N0*bw; % total noise power in watts
nVar = nPower./2; % noise variance, frequency domain

% Generate channel covariance matrix
Rhh = getRhh(numPath,numSC,1e5);

% Testing stage
ITER = 20; % Increase iterations for better statistics
numErr_SIC_LS_16 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_SIC_MMSE_16 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_ML_16 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL_16 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_SIC_LS_64 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_SIC_MMSE_64 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_ML_64 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL_64 = zeros(numUE,numel(EsN0_dB),ITER);

tic;
for it = 1:ITER
    for snr = 1:numel(nVar)
        % Set random seed for reproducibility in testing
        s = RandStream('mt19937ar','Seed',it+snr); % Different seed for each iteration and SNR
        RandStream.setGlobalStream(s);

        % Transmit packets for 16 pilots
        pilotFrame_16 = zeros(numPSym,numSC,numUE,numPacket);
        pilotFrame_16(1,:,1,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
        pilotFrame_16(2,:,2,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
        pilotFrame_16(:,pilotStart(1):pilotSpacing_16:end,1,:) = fixedPilot_16(:,:,1,:);
        pilotFrame_16(:,pilotStart(2):pilotSpacing_16:end,2,:) = fixedPilot_16(:,:,2,:);
        
        % Transmit packets for 64 pilots
        pilotFrame_64 = zeros(numPSym,numSC,numUE,numPacket);
        pilotFrame_64(1,:,1,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
        pilotFrame_64(2,:,2,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
        pilotFrame_64(:,pilotStart(1):pilotSpacing_64:end,1,:) = fixedPilot_64(:,:,1,:);
        pilotFrame_64(:,pilotStart(2):pilotSpacing_64:end,2,:) = fixedPilot_64(:,:,2,:);
        
        dataFrame = complex(sign(rand(numDSym,numSC,numUE,numPacket)-0.5),sign(rand(numDSym,numSC,numUE,numPacket)-0.5));
        transmitPacket_16 = zeros(numSym,numSC,numUE,numPacket);
        transmitPacket_16(1:2,:,:,:) = pilotFrame_16;
        transmitPacket_16(end,:,:,:) = 1/sqrt(2)*dataFrame;
        transmitPacket_64 = zeros(numSym,numSC,numUE,numPacket);
        transmitPacket_64(1:2,:,:,:) = pilotFrame_64;
        transmitPacket_64(end,:,:,:) = 1/sqrt(2)*dataFrame;
        
        % Collect labels for transmitted data symbols
        tLabel = zeros(1,numPacket);
        for b = 1:numel(labelClass)
            tLabel(logical(squeeze(dataFrame(1,idx_sc,1,:))==symClass(b,1) & squeeze(dataFrame(1,idx_sc,2,:))==symClass(b,2))) = b;
        end
        
        % Allocate power
        [powerFactor,decOrder] = allocatePower(symPower,gainH,targetSNR_linear_1,targetSNR_linear_2,nVar(snr));
        h_all = repmat(h,1,1,numPacket);
        powerFactor_all = repmat(powerFactor,1,1,numPacket);
        decOrder_all = repmat(decOrder,1,1,numPacket);
        
        % Received packets with random phase for 16 pilots
        [receivePacket_16,randomPhase] = dataTransmissionReception(transmitPacket_16,powerFactor_all,lengthCP,h_all,nVar(snr));
        receivePilot_16 = receivePacket_16(1:2,:,:); 
        receiveData_16 = receivePacket_16(end,:,:);
        
        % Received packets with random phase for 64 pilots
        [receivePacket_64,randomPhase] = dataTransmissionReception(transmitPacket_64,powerFactor_all,lengthCP,h_all,nVar(snr));
        receivePilot_64 = receivePacket_64(1:2,:,:); 
        receiveData_64 = receivePacket_64(end,:,:);
        
        % ML detection (with perfect CSI) for 16 pilots
        decOrder_sc = squeeze(decOrder_all(idx_sc,:,:));
        idx_1 = decOrder_sc(1,:).';
        idx_2 = decOrder_sc(2,:).';
        H_sc = repmat(H(idx_sc,:),numPacket,1).';
        pF_sc = squeeze(powerFactor_all(idx_sc,:,:)); 
        rData_16 = squeeze(receiveData_16(1,idx_sc,:)); % numPacket x 1
        tData = squeeze(transmitPacket_16(end,idx_sc,:,:)); % numUE x numPacket
        [numErr_ML_16(:,snr,it),~] = detectML(H_sc,randomPhase,constQPSK,pF_sc,rData_16,idx_1,idx_2,tData,symClass);
        
        % ML detection (with perfect CSI) for 64 pilots
        rData_64 = squeeze(receiveData_64(1,idx_sc,:)); % numPacket x 1
        [numErr_ML_64(:,snr,it),~] = detectML(H_sc,randomPhase,constQPSK,pF_sc,rData_64,idx_1,idx_2,tData,symClass);
        
        % LS and MMSE estimation with SIC for 16 pilots
        [H_LS_16,H_MMSE_16] = channelEstimation(receivePilot_16,pilotFrame_16,powerFactor_all,pilotStart,Rhh,nVar(snr),numPSC_16pilots,H);
        [numErr_SIC_LS_16(:,snr,it),~] = symbolDecodeSIC(rData_16,H_LS_16(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
        [numErr_SIC_MMSE_16(:,snr,it),~] = symbolDecodeSIC(rData_16,H_MMSE_16(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
        
        % LS and MMSE estimation with SIC for 64 pilots
        [H_LS_64,H_MMSE_64] = channelEstimation(receivePilot_64,pilotFrame_64,powerFactor_all,pilotStart,Rhh,nVar(snr),numPSC_64pilots,H);
        [numErr_SIC_LS_64(:,snr,it),~] = symbolDecodeSIC(rData_64,H_LS_64(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
        [numErr_SIC_MMSE_64(:,snr,it),~] = symbolDecodeSIC(rData_64,H_MMSE_64(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
        
        % DL detection for 16 pilots
        [numErr_DL_16(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_16,tLabel,net_16pilots,decOrder_sc,symClass,constQPSK);
        
        % DL detection for 64 pilots
        [numErr_DL_64(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_64,tLabel,net_64pilots,decOrder_sc,symClass,constQPSK);
    end
end
toc;

% Average over iterations
numErr_SIC_LS_16 = mean(numErr_SIC_LS_16,3);
numErr_SIC_MMSE_16 = mean(numErr_SIC_MMSE_16,3);
numErr_ML_16 = mean(numErr_ML_16,3);
numErr_DL_16 = mean(numErr_DL_16,3);
numErr_SIC_LS_64 = mean(numErr_SIC_LS_64,3);
numErr_SIC_MMSE_64 = mean(numErr_SIC_MMSE_64,3);
numErr_ML_64 = mean(numErr_ML_64,3);
numErr_DL_64 = mean(numErr_DL_64,3);

% Plot SER for User 1 with 16 and 64 pilots (Fig. 5)
figure();
semilogy(EsN0_dB, numErr_SIC_LS_16(1,:), 'b-o', 'LineWidth', 1.5, 'DisplayName', '16 Pilots, SIC with LS'); hold on;
semilogy(EsN0_dB, numErr_SIC_MMSE_16(1,:), 'k-o', 'LineWidth', 1.5, 'DisplayName', '16 Pilots, SIC with MMSE'); hold on;
semilogy(EsN0_dB, numErr_DL_16(1,:), 'r-o', 'LineWidth', 1.5, 'DisplayName', '16 Pilots, DL'); hold on;
semilogy(EsN0_dB, numErr_SIC_LS_64(1,:), 'b--o', 'LineWidth', 1.5, 'DisplayName', '64 Pilots, SIC with LS'); hold on;
semilogy(EsN0_dB, numErr_SIC_MMSE_64(1,:), 'k--o', 'LineWidth', 1.5, 'DisplayName', '64 Pilots, SIC with MMSE'); hold on;
semilogy(EsN0_dB, numErr_DL_64(1,:), 'r--o', 'LineWidth', 1.5, 'DisplayName', '64 Pilots, DL'); hold off;
title('SER curves for User 1 with 16 and 64 Pilot Symbols');
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
legend('show');
grid on;

% Save the plot
saveas(gcf, 'Figure5_SER_vs_SNR_User1_16_64pilots.png');

function Rhh = getRhh(numPaths, numSC, numChan)
    Rhh = zeros(numSC, numSC, numChan);
    for i = 1:numChan    
        h = 1/sqrt(2)/sqrt(numPaths)*complex(randn(numPaths,1),randn(numPaths,1)); % L x 1
        H = fft(h, numSC); % numSC x 1
        Rhh(:,:,i) = H*H';    
    end
    Rhh = mean(Rhh, 3); % numSC x numSC
end