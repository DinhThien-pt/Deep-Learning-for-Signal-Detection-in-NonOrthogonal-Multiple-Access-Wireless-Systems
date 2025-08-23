clear variables;
close all;

% Random seed for reproducing static channel and fixed training data
s = RandStream('mt19937ar','Seed',1921164231);
RandStream.setGlobalStream(s);

% System parameters
lengthCP = 12; % length of cyclic prefix (for CP=12)
numPSC_16pilots = 16; % number of pilot subcarriers (as in Fig. 5)
numUE = 2;
numSC = 64; % number of subcarriers
numPSym = numUE; % number of pilot OFDM symbols per packet
numDSym = 1; % number of data OFDM symbol per packet
numSym = numPSym+numDSym; % number of OFDM symbols per packet
pilotSpacing = numSC/numPSC_16pilots;
pilotStart = [1,1]; % pilot starting subcarrier for two users 

% Data symbol modulation
constQPSK = [1-1j;1+1j;-1+1j;-1-1j];
a = constQPSK(1);
b = constQPSK(2);
c = constQPSK(3);
d = constQPSK(4);
% Symbol combination class
symComb = [a a;a b;a c;a d;b a;b b;b c;b d;c a;c b;c c;c d;d a;d b;d c;d d]; 
labelClass = 1:1:size(symComb,1);
numLabel = length(labelClass);

% Noise computation
EsN0_dB = 40;
EsN0 = 10.^(EsN0_dB./10);
symRate = 2; % symbol rate, 2 symbol/s
Es = 1; % symbol energy, joules/symbol
sigPower = Es*symRate; % total signal power, watts
symPower = sigPower/numUE; % signal power per symbol 
N0 = sigPower./EsN0; % noise power in watts/Hz
bw = 1; % bandwidth per subcarrier, Hz
nPower = N0*bw; % total noise power in watts
nVar = nPower./2; % noise variance, frequency domain

% Power allocation in frequency domain
targetSNR_1 = 12; % dB, target SNR for strong user
targetSNR_2 = 12; % dB, target SNR for weak user
targetSNR_linear_1 = 10^(targetSNR_1/10);
targetSNR_linear_2 = 10^(targetSNR_2/10);
% Static channel realisation
numPath = 20;
h = 1/sqrt(2)/sqrt(numPath)*complex(randn(numPath,numUE),randn(numPath,numUE));
H = fft(h,numSC,1); 
gainH = (abs(H).^2).';
% Calculate power allocation factor and obtain decoding order
[powerFactor,decOrder] = allocatePower(symPower,gainH,targetSNR_linear_1,targetSNR_linear_2,nVar);

% Training data generation
numPacketClass = 3e4; % number of OFDM packets per label (480,000 total)
% Fixed pilot symbols (BPSK modulation)
fixedPilot_16pilots = zeros(numPSym,numPSC_16pilots,numUE);
fixedPilot_16pilots(1,:,1) = complex(sign(rand(1,numPSC_16pilots,1)-0.5)); 
fixedPilot_16pilots(2,:,2) = complex(sign(rand(1,numPSC_16pilots,1)-0.5));
fixedPilotPacket = repmat(fixedPilot_16pilots,1,1,1,numPacketClass); % use the same pilot for all packets
% Target subcarrier for signal detection
idx_sc = 20; 
XTrain = []; % training samples
YTrain = []; % labels

tic;
for n = 1:numLabel % generate training data for each class
    % Pilot symbols (fixed with seed)
    pilotFrame = zeros(numPSym,numSC,numUE,numPacketClass);
    pilotFrame(1,:,1,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacketClass)-0.5),sign(rand(1,numSC,1,numPacketClass)-0.5));
    pilotFrame(2,:,2,:) = 1/sqrt(2)*complex(sign(rand(1,numSC,1,numPacketClass)-0.5),sign(rand(1,numSC,1,numPacketClass)-0.5)); 
    pilotFrame(:,pilotStart(1):pilotSpacing:end,1,:) = fixedPilotPacket(:,:,1,:);
    pilotFrame(:,pilotStart(2):pilotSpacing:end,2,:) = fixedPilotPacket(:,:,2,:);

    % Data symbols (fixed with seed)
    dataFrame = 1/sqrt(2)*complex(sign(rand(numDSym,numSC,numUE,numPacketClass)-0.5),sign(rand(numDSym,numSC,numUE,numPacketClass)-0.5));
    % Replace random data symbols with current data combination on the target subcarrier
    currentData = repmat(symComb(n,:),1,1,numPacketClass); 
    currentData = reshape(currentData,1,1,numUE,numPacketClass); 
    dataFrame(:,idx_sc,:,:) = 1/sqrt(2)*currentData;
    
    % Data transmission and reception
    hAll = repmat(h,1,1,numPacketClass);
    powerFactorAll = repmat(powerFactor,1,1,numPacketClass);
    decOrderAll = repmat(decOrder,1,1,numPacketClass);
    transmitPacket = zeros(numSym,numSC,numUE,numPacketClass);
    transmitPacket(1:2,:,:,:) = pilotFrame;
    transmitPacket(end,:,:,:) = dataFrame;
    [receivePacket,~] = dataTransmissionReception(transmitPacket,powerFactorAll,lengthCP,hAll,nVar);
    
    % Construct feature vector and labels
    dataLabel = n*ones(1,numPacketClass);
    [feature,label,~] = getFeatureAndLabel(real(receivePacket),imag(receivePacket),dataLabel,n);
    featureVec = mat2cell(feature,size(feature,1),ones(1,size(feature,2))); % cell, 1 x #perClass, each cell, 384 x 1
    XTrain = [XTrain featureVec];
    YTrain = [YTrain label];
end
toc;

XTrain = XTrain.';
YTrain = categorical(YTrain.');

save('trainDataCP12_16pilots.mat','XTrain','YTrain','h','numPSC_16pilots','lengthCP','idx_sc','fixedPilot_16pilots');