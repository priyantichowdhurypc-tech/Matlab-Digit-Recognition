clc;
clear;
close all;

%% STEP 1: Load Dataset & Extract Labels from Filenames
digitDatasetPath = fullfile('DigitRecognitionProject');

if ~exist(digitDatasetPath, 'dir')
    error("The folder '" + digitDatasetPath + "' does not exist. Please check your path!");
end

imdsAll = imageDatastore(fullfile(digitDatasetPath, '*.png'));
[~, fileNames, ~] = cellfun(@fileparts, imdsAll.Files, 'UniformOutput', false);

extractedLabels = cellfun(@(x) x(1), fileNames, 'UniformOutput', false);
imdsAll.Labels = categorical(extractedLabels);

binaryLabels = {'0', '1'}; 
isZeroOrOne = ismember(string(imdsAll.Labels), binaryLabels);

imds = subset(imdsAll, find(isZeroOrOne));
imds.Labels = categorical(string(imds.Labels)); 

numClasses = numel(categories(imds.Labels));
disp("Training model to recognize " + numClasses + " classes: " + strjoin(categories(imds.Labels), ', '));
tabulate(imds.Labels);

%% STEP 2: Split Data
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

%% STEP 3: Pure Custom Preprocessing (Guarantees Training matches Prediction)
% We attach a manual ReadFcn to the datastore. Every image trained on will now 
% pass through the exact same adaptive binarization and inversion process.
imdsTrain.ReadFcn = @(loc) preprocessPipeline(loc);
imdsTest.ReadFcn  = @(loc) preprocessPipeline(loc);

inputSize = [28 28 1]; 

augmenter = imageDataAugmenter(...
    'RandXTranslation', [-2 2], ...
    'RandYTranslation', [-2 2], ...
    'RandRotation', [-10 10]);

% We drop ColorPreprocessing because our function already handles grayscale output
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', augmenter);
augImdsTest = augmentedImageDatastore(inputSize, imdsTest);

%% STEP 4: CNN Model
layers = [
    imageInputLayer(inputSize)
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
    
    fullyConnectedLayer(numClasses) 
    softmaxLayer
    classificationLayer
    ];

%% STEP 5: Training Options
options = trainingOptions('sgdm', ...
    'MaxEpochs', 15, ... % Raised epochs to let weights stabilize perfectly
    'InitialLearnRate', 0.01, ...
    'Verbose', false, ...
    'Plots','training-progress');

%% STEP 6: Train Network using Augmented Data
net = trainNetwork(augImdsTrain, layers, options);

%% STEP 7: Verify Dataset Test Accuracy
YPred = classify(net, augImdsTest);
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest);
disp("Dataset Validation Accuracy = " + (accuracy * 100) + "%");

%% STEP 8: Predict Your Custom Images
files = dir(fullfile(digitDatasetPath, '*.png'));

if isempty(files)
    disp("No image files found! Please check your folder path.");
else
    fprintf('\n--- Individual Predictions ---\n');
    for i = 1:length(files)
        imgpath = fullfile(digitDatasetPath, files(i).name);
        
        % Call the exact same processing function used during training
        img_final = preprocessPipeline(imgpath);
        
        % Reshape to 3D Tensor format for individual classification
        img_final = reshape(img_final, [28, 28, 1]);
        
        label = classify(net, img_final);
        disp(files(i).name + " -> Predicted Label: " + string(label));
    end
end

%% --- HELPER FUNCTION FOR EXACT MATCHING PREPROCESSING ---
function out = preprocessPipeline(filename)
    img = imread(filename);
    
    if size(img, 3) == 3
        img = im2gray(img);
    end
    
    % 1. Force background to crisp binary separation
    img_bin = imbinarize(img, 'adaptive', 'Sensitivity', 0.5);
    
    % 2. Flip so the background is pure 0 (black) and ink is 1 (white)
    img_inv = ~img_bin; 
    
    % 3. Resize and cast precision
    img_final = imresize(img_inv, [28 28]);
    out = im2single(img_final);
end