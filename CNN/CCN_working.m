function accuracy=CCN_working()
digitDatasetPath ='C:\Users\ilya\North Carolina State University\CSS522 - Documents\project_data1';

imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');
%figure;imshow(imds.Files{2});
%im1=imread(imds.Files{2});
numTrainingFiles = 100;
[imdsTrain,imdsTest] = splitEachLabel(imds,numTrainingFiles,'randomize');
options = trainingOptions('sgdm', ...
    'MaxEpochs',120,...
    'InitialLearnRate',1e-4, ...
    'ValidationData',imdsTest, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');


layers = [ ...
    imageInputLayer([16 90 1])
    convolution2dLayer([1 5],40)%[1 4],16
    reluLayer
    maxPooling2dLayer([1 5],'Stride',[1 5])
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

net = trainNetwork(imdsTrain,layers,options);

YPred = classify(net,imdsTest);
YTest = imdsTest.Labels;
accuracy = sum(YPred == YTest)/numel(YTest);
end


