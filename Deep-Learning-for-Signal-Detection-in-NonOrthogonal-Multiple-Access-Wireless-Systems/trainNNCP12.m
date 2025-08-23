clear variables;
close all;

% Load training data and essential parameters
load('trainDataCP12.mat','XTrain','YTrain');

numSC = 64;

% Batch size
miniBatchSize = 20000; % as in Table I

% Iteration
maxEpochs = 100; % as in Table I

% Structure
inputSize = 2 * numSC * 3; % 384
numHiddenUnits = 128; 
numHiddenUnits2 = 64;
numHiddenUnits3 = numSC;
numClasses = 16;

% DNN Layers
layers = [ ...
    sequenceInputLayer(inputSize, 'Name', 'input')
    flattenLayer('Name', 'flatten')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last', 'Name', 'lstm')
    fullyConnectedLayer(numClasses, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classoutput')];

% Split data into training and validation sets (80-20 split)
numObservations = numel(XTrain);
numTrain = floor(0.8 * numObservations);
idx = randperm(numObservations);
idxTrain = idx(1:numTrain);
idxVal = idx(numTrain+1:end);

XTrain_new = XTrain(idxTrain);
YTrain_new = YTrain(idxTrain);
XVal = XTrain(idxVal);
YVal = YTrain(idxVal);

% Training options
options = trainingOptions('adam',...
    'InitialLearnRate', 0.01,... % as in Table I
    'ExecutionEnvironment', 'auto', ...
    'GradientThreshold', 1, ...
    'LearnRateDropFactor', 0.1,...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 1,...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 30, ... % Adjust as needed based on dataset size
    'Plots', 'training-progress'); 

% Train the neural network
tic;
net = trainNetwork(XTrain_new, YTrain_new, layers, options);
toc;

save('NNCP12.mat', 'net');