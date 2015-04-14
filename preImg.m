function result = preImg(file)
    %% read the image file
    if strcmp(file(end-2:end), 'png')
        [img, map] = imread(file);
        img = ind2gray(img,map);
    else
        img = imread(file);
        img = rgb2gray(img);
    end
    %% preprocess the image
    % turn to binary image
    img = im2bw(img);
    % make the background black and numbers white
    if sum(img(:)) > numel(img)*0.5
        img = ~img;
    end
    % eliminate small objects on the image
    img = bwareaopen(img,100);
    
    %% use connected component of a graph to find the segmentation
    s = regionprops(img, 'BoundingBox');
    result = zeros(50, 50, length(s));
    figure;
    for i = 1:length(s)
        component = int32(s(i).BoundingBox);
        separateImg = img(component(2):component(2)+component(4), component(1):component(1)+component(3));
        Imgback = 255 * uint8(separateImg);
        separateImgback = makeRec(Imgback);
        result(:, :, i) = imresize(separateImgback, [50 50], 'nearest');
    end
    % add edge to every component of the image and reszie it to 28*28 to
    % fit into neural network prediction
    result = addedge(result);
    %% show the figures of an image
    for i = 1: length(s)
        subplot(5, 5, i);
        tempImg = result(:, :, i);
        imshow(tempImg);
    end
end

function img = makeRec(img)
    % turn the image into a rectangular image by adding zeros
    % the length of the sides is decided by the longer one
    [rowNum, columnNum] = size(img);
    zeroNum = round(abs(rowNum-columnNum)/2);
    
    if rowNum > columnNum
        img = [zeros(rowNum,zeroNum) img zeros(rowNum,zeroNum)];
    else
        img = [zeros(zeroNum,columnNum); img; zeros(zeroNum,columnNum)];
    end
end

function newImgs = addedge(imgs)
    % add an 50% margin to each side of the image
    zeroline = zeros(15, 50);
    zerocolumn = zeros(80, 15);
    newImgs = zeros(28, 28, size(imgs,3));
    for i = 1:size(imgs, 3)
        temp = [zeroline;imgs(:,:,i);zeroline];
        temp = abs(imresize([zerocolumn temp zerocolumn], [28 28]));
        %normalize to match the training data
        newImgs(:,:,i) = sqrt(temp/(max(temp(:)) - min(temp(:))));
    end
end   
     
     
    