function preds = recognize(images)
addpath ./ConvolutionalNeuralNetwork

load cnnPara;
labels = 0;
[~,~,preds]=cnnCost(opttheta,images,labels,numClasses,...
                filterDim,numFilters,poolDim,true);
end