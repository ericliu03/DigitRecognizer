function preds = recognize(images)
addpath ./ConvolutionalNeuralNetwork

load cnnPara;
labels = ones(1, size(images,3))*10;

[~,~,preds]=cnnCost(opttheta,images,labels,numClasses,...
                filterDim,numFilters,poolDim,true);
end