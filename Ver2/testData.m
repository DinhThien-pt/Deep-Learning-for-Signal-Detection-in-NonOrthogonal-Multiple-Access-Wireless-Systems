clear variables;
close all;

% Load training data and essential parameters
load('trainDataCP20.mat','h','numPSC','idx_sc','fixedPilot');
% Load neural networks
load('NNCP20.mat','net'); % Network for CP=20
netCP20 = net;
load('NNCP12.mat','net'); % Network for CP=12
netCP12 = net;

% System parameters
[numPath,numUE] = size(h);
numSC = 64; % number of subcarriers
numPSym = numUE; % number of pilot OFDM symbols per packet
numDSym = 1; % number of data OFDM symbol per packet
numSym = numPSym+numDSym; % number of OFDM symbols per packet
pilotSpacing = numSC/numPSC;
pilotStart = [1,1]; % pilot starting subcarrier for two users 
lengthCP_list = [20, 12]; % CP lengths to test (as in Fig. 3)

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
fixedPilot = repmat(fixedPilot,1,1,1,numPacket);

% Power allocations
targetSNR_1 = 12; % dB, target SNR for strong user
targetSNR_2 = 12; % dB, target SNR for weak user
targetSNR_linear_1 = 10^(targetSNR_1/10);
targetSNR_linear_2 = 10^(targetSNR_2/10);
H = fft(h,numSC,1); 
gainH = (abs(H).^2).';

% Noise computation
EsN0_dB = 0:2:26; % Start from 0 dB to ensure high noise at low SNR
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
numErr_SIC_LS = zeros(numUE,numel(EsN0_dB),ITER,length(lengthCP_list));
numErr_SIC_MMSE = zeros(numUE,numel(EsN0_dB),ITER,length(lengthCP_list));
numErr_ML = zeros(numUE,numel(EsN0_dB),ITER,length(lengthCP_list));
numErr_DL = zeros(numUE,numel(EsN0_dB),ITER,length(lengthCP_list));

tic;
for cp_idx = 1:length(lengthCP_list)
    lengthCP_current = lengthCP_list(cp_idx);
    for it = 1:ITER
        for snr = 1:numel(nVar)
            % Set random seed for reproducibility in testing
            s = RandStream('mt19937ar','Seed',it+snr); % Different seed for each iteration and SNR
            RandStream.setGlobalStream(s);

            % Transmit packets
            pilotFrame = zeros(numPSym,numSC,numUE,numPacket);
            pilotFrame(1,:,1,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
            pilotFrame(2,:,2,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacket)-0.5),sign(rand(1,numSC,1,numPacket)-0.5));
            pilotFrame(:,pilotStart(1):pilotSpacing:end,1,:) = fixedPilot(:,:,1,:);
            pilotFrame(:,pilotStart(2):pilotSpacing:end,2,:) = fixedPilot(:,:,2,:);
            dataFrame = complex(sign(rand(numDSym,numSC,numUE,numPacket)-0.5),sign(rand(numDSym,numSC,numUE,numPacket)-0.5));
            transmitPacket = zeros(numSym,numSC,numUE,numPacket);
            transmitPacket(1:2,:,:,:) = pilotFrame;
            transmitPacket(end,:,:,:) = 1/sqrt(2)*dataFrame;
            
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
            
            % Received packets with random phase
            [receivePacket,randomPhase] = dataTransmissionReception(transmitPacket,powerFactor_all,lengthCP_current,h_all,nVar(snr));
            receivePilot = receivePacket(1:2,:,:); 
            receiveData = receivePacket(end,:,:);
            
            % Debug: Check received data at low SNR
            if snr == 1 && it == 1 && cp_idx == 1
                disp('Received data at SNR 0 dB (real):');
                disp(real(receiveData(1,idx_sc,1)));
            end
            
            % ML detection (with perfect CSI)
            decOrder_sc = squeeze(decOrder_all(idx_sc,:,:));
            idx_1 = decOrder_sc(1,:).';
            idx_2 = decOrder_sc(2,:).';
            H_sc = repmat(H(idx_sc,:),numPacket,1).';
            pF_sc = squeeze(powerFactor_all(idx_sc,:,:)); 
            rData = squeeze(receiveData(1,idx_sc,:)); % numPacket x 1
            tData = squeeze(transmitPacket(end,idx_sc,:,:)); % numUE x numPacket
            [numErr_ML(:,snr,it,cp_idx),~] = detectML(H_sc,randomPhase,constQPSK,pF_sc,rData,idx_1,idx_2,tData,symClass);
            
            % LS and MMSE estimation with SIC
            [H_LS,H_MMSE] = channelEstimation(receivePilot,pilotFrame,powerFactor_all,pilotStart,Rhh,nVar(snr),numPSC,H);
            [numErr_SIC_LS(:,snr,it,cp_idx),~] = symbolDecodeSIC(rData,H_LS(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
            [numErr_SIC_MMSE(:,snr,it,cp_idx),~] = symbolDecodeSIC(rData,H_MMSE(idx_sc,:,:),decOrder_sc,pF_sc,constQPSK,tData);
            
            % DL detection (use appropriate network based on CP length)
            if cp_idx == 1
                net = netCP20; % Use network trained for CP=20
            else
                net = netCP12; % Use network trained for CP=12
            end
            [numErr_DL(:,snr,it,cp_idx), ~] = symbolDecodeDL(labelClass,receivePacket,tLabel,net,decOrder_sc,symClass,constQPSK);
        end
    end
end
toc;

% Average over iterations
numErr_SIC_LS = mean(numErr_SIC_LS,3);
numErr_SIC_MMSE = mean(numErr_SIC_MMSE,3);
numErr_ML = mean(numErr_ML,3);
numErr_DL = mean(numErr_DL,3);

% Plot SER for User 1 (Fig. 3)
figure();
semilogy(EsN0_dB, numErr_SIC_LS(1,:,1,1), 'b-o', 'LineWidth', 1.5, 'DisplayName', 'SIC with LS, CP=20'); hold on;
semilogy(EsN0_dB, numErr_SIC_LS(1,:,1,2), 'b--o', 'LineWidth', 1.5, 'DisplayName', 'SIC with LS, CP=12'); hold on;
semilogy(EsN0_dB, numErr_SIC_MMSE(1,:,1,1), 'k-o', 'LineWidth', 1.5, 'DisplayName', 'SIC with MMSE, CP=20'); hold on;
semilogy(EsN0_dB, numErr_SIC_MMSE(1,:,1,2), 'k--o', 'LineWidth', 1.5, 'DisplayName', 'SIC with MMSE, CP=12'); hold on;
semilogy(EsN0_dB, numErr_ML(1,:,1,1), 'g-o', 'LineWidth', 1.5, 'DisplayName', 'ML with perfect CSI, CP=20'); hold on;
semilogy(EsN0_dB, numErr_ML(1,:,1,2), 'g--o', 'LineWidth', 1.5, 'DisplayName', 'ML with perfect CSI, CP=12'); hold on;
semilogy(EsN0_dB, numErr_DL(1,:,1,1), 'r-o', 'LineWidth', 1.5, 'DisplayName', 'DL, CP=20'); hold on;
semilogy(EsN0_dB, numErr_DL(1,:,1,2), 'r--o', 'LineWidth', 1.5, 'DisplayName', 'DL, CP=12'); hold off;
title('SER for User 1 with Different CP Lengths');
xlabel('SNR (dB)');
ylabel('Symbol Error Rate (SER)');
legend('show');
grid on;

% Save the plot
saveas(gcf, 'Figure3_SER_vs_SNR.png');

function Rhh = getRhh(numPaths, numSC, numChan)
    Rhh = zeros(numSC, numSC, numChan);
    for i = 1:numChan    
        h = 1/sqrt(2)/sqrt(numPaths)*complex(randn(numPaths,1),randn(numPaths,1)); % L x 1
        H = fft(h, numSC); % numSC x 1
        Rhh(:,:,i) = H*H';    
    end
    Rhh = mean(Rhh, 3); % numSC x numSC
end