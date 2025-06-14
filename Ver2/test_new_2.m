clear variables;
close all;

% Load training data and essential parameters
load('trainDataCP12_16pilots.mat','h','numPSC_16pilots','idx_sc','fixedPilot_16pilots');
load('trainDataCP12_64pilots.mat','h','numPSC_64pilots','idx_sc','fixedPilot_64pilots');

% Kiểm tra sự nhất quán của h và idx_sc
if ~isequal(h, h) || ~isequal(idx_sc, idx_sc)
    error('Biến h hoặc idx_sc không nhất quán giữa 16pilots và 64pilots.');
end

% Load neural networks with different batch sizes for 16 pilots
load('NNCP12_16pilots_bs2000.mat','net'); net_bs2000_16 = net;
load('NNCP12_16pilots_bs5000.mat','net'); net_bs5000_16 = net;
load('NNCP12_16pilots_bs10000.mat','net'); net_bs10000_16 = net;
load('NNCP12_16pilots_bs20000.mat','net'); net_bs20000_16 = net;

% Load neural networks with different batch sizes for 64 pilots (optional)
load('NNCP12_64pilots_bs2000.mat','net'); net_bs2000_64 = net;
load('NNCP12_64pilots_bs5000.mat','net'); net_bs5000_64 = net;
load('NNCP12_64pilots_bs10000.mat','net'); net_bs10000_64 = net;
load('NNCP12_64pilots_bs20000.mat','net'); net_bs20000_64 = net;

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
EsN0_dB = 0:2:28; % Start from 0 dB to match Fig. 8 range
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
Rhh = getRhh(numPath, numSC, 1e5);

% Testing stage
ITER = 20; % Increase iterations for better statistics
% For 16 pilots
numErr_DL_16_bs2000 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL_16_bs5000 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL_16_bs10000 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL_16_bs20000 = zeros(numUE,numel(EsN0_dB),ITER);
% For 64 pilots (optional)
numErr_DL_64_bs2000 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL_64_bs5000 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL_64_bs10000 = zeros(numUE,numel(EsN0_dB),ITER);
numErr_DL_64_bs20000 = zeros(numUE,numel(EsN0_dB),ITER);

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
        
        % DL detection for 16 pilots with different batch sizes
        decOrder_sc = squeeze(decOrder_all(idx_sc,:,:));
        [numErr_DL_16_bs2000(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_16,tLabel,net_bs2000_16,decOrder_sc,symClass,constQPSK);
        [numErr_DL_16_bs5000(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_16,tLabel,net_bs5000_16,decOrder_sc,symClass,constQPSK);
        [numErr_DL_16_bs10000(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_16,tLabel,net_bs10000_16,decOrder_sc,symClass,constQPSK);
        [numErr_DL_16_bs20000(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_16,tLabel,net_bs20000_16,decOrder_sc,symClass,constQPSK);
        
        % DL detection for 64 pilots with different batch sizes (optional)
        [numErr_DL_64_bs2000(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_64,tLabel,net_bs2000_64,decOrder_sc,symClass,constQPSK);
        [numErr_DL_64_bs5000(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_64,tLabel,net_bs5000_64,decOrder_sc,symClass,constQPSK);
        [numErr_DL_64_bs10000(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_64,tLabel,net_bs10000_64,decOrder_sc,symClass,constQPSK);
        [numErr_DL_64_bs20000(:,snr,it), ~] = symbolDecodeDL(labelClass,receivePacket_64,tLabel,net_bs20000_64,decOrder_sc,symClass,constQPSK);
    end
end
toc;

% Average over iterations
numErr_DL_16_bs2000 = mean(numErr_DL_16_bs2000,3);
numErr_DL_16_bs5000 = mean(numErr_DL_16_bs5000,3);
numErr_DL_16_bs10000 = mean(numErr_DL_16_bs10000,3);
numErr_DL_16_bs20000 = mean(numErr_DL_16_bs20000,3);
numErr_DL_64_bs2000 = mean(numErr_DL_64_bs2000,3);
numErr_DL_64_bs5000 = mean(numErr_DL_64_bs5000,3);
numErr_DL_64_bs10000 = mean(numErr_DL_64_bs10000,3);
numErr_DL_64_bs20000 = mean(numErr_DL_64_bs20000,3);

% Plot SER curves for DNN trained with different batch sizes (Fig. 8)
figure();
semilogy(EsN0_dB, numErr_DL_16_bs2000(1,:), 'g-d', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'User 1, Batch size = 2000'); hold on;
semilogy(EsN0_dB, numErr_DL_16_bs5000(1,:), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'User 1, Batch size = 5000'); hold on;
semilogy(EsN0_dB, numErr_DL_16_bs10000(1,:), 'r-*', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'User 1, Batch size = 10000'); hold on;
semilogy(EsN0_dB, numErr_DL_16_bs20000(1,:), 'k-d', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'User 1, Batch size = 20000'); hold on;
semilogy(EsN0_dB, numErr_DL_16_bs2000(2,:), 'g-d', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'User 2, Batch size = 2000'); hold on;
semilogy(EsN0_dB, numErr_DL_16_bs5000(2,:), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'User 2, Batch size = 5000'); hold on;
semilogy(EsN0_dB, numErr_DL_16_bs10000(2,:), 'r-*', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'User 2, Batch size = 10000'); hold on;
semilogy(EsN0_dB, numErr_DL_16_bs20000(2,:), 'k-d', 'LineWidth', 1.5, 'MarkerSize', 6, 'DisplayName', 'User 2, Batch size = 20000'); hold off;
title('SER curves of DNN trained with different batch sizes');
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
legend('show');
grid on;

% Save the plot
saveas(gcf, 'Figure8_SER_vs_SNR_DNN_BatchSizes.png');

% Function to generate channel covariance matrix
function Rhh = getRhh(numPaths, numSC, numChan)
    Rhh = zeros(numSC, numSC, numChan);
    for i = 1:numChan    
        h = 1/sqrt(2)/sqrt(numPaths)*complex(randn(numPaths,1),randn(numPaths,1)); % L x 1
        H = fft(h, numSC); % numSC x 1
        Rhh(:,:,i) = H*H';    
    end
    Rhh = mean(Rhh, 3); % numSC x numSC
end