%% test numbers from MNIST
addpath './ConvolutionalNeuralNetwork';
testImages = loadMNISTImages('trainingData/t10k-images-idx3-ubyte');
testLabels = loadMNISTLabels('trainingData/t10k-labels-idx1-ubyte');

imageDim = 28;
testImages = reshape(testImages,imageDim,imageDim,[]);

testNum = 18;
randIndices = randi(10000,1,testNum);


demoLabels = zeros(testNum);
demoImages = zeros(imageDim,imageDim,testNum);
for i = 1:testNum
    demoImages(:,:,i) = testImages(:,:,randIndices(i));
    demoLabels(i) = testLabels(randIndices(i));
end

pred = recognize(demoImages);
pred(pred==10) = 0;

figure;
for i = 1:length(pred)
    subplot(3,6,i);
    imshow(demoImages(:,:,i));
    if pred(i) == demoLabels(i)
        xlabel(pred(i),'FontSize',14);
    else
        xlabel(pred(i),'Color','r','FontSize',14);
    end
end


%% self-made numbers
file = 'test2.jpg';
img = preImg(file);
pred = recognize(img);
pred(pred==10) = 0;
imgNum = size(img,3);
figure;
subplot(2,imgNum,[1 imgNum]);
imshow(imread(file));
for i = 1:imgNum
    subplot(2,imgNum,imgNum + i);
    imshow(img(:,:,i));
    xlabel(pred(i),'FontSize',14);
end
disp('prediction is :');
fprintf('%d', pred);
disp(' ');