clear variables;
close all;

% Load training data and essential parameters
load('trainDataCP12_16pilots.mat','XTrain','YTrain');

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

% Training options
options = trainingOptions('adam',...
    'InitialLearnRate', 0.001,... % Changed to 0.001
    'ExecutionEnvironment', 'auto', ...
    'GradientThreshold', 1, ...
    'LearnRateDropFactor', 0.1,...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', 1,...
    'Plots', 'training-progress'); 

% Train the neural network
tic;
net = trainNetwork(XTrain, YTrain, layers, options);
toc;

save('NNCP12_16pilots_lr0001.mat', 'net');