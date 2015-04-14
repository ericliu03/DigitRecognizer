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
    % eliminate the noises
    img = bwareaopen(img,4);
    % turn to gray scale
    imgback = 255* uint8(img);
    % imshow(imgback);

    %% find the segmentation
    %compute projection in one dimension
    project_x = sum(img);
    seg_x = findEdge(project_x);

    result = zeros(50, 50, length(seg_x)/2);

    % seperate each charactor
    for i = 1:length(seg_x)/2
       temp = imgback(:,seg_x(2*i-1)+1:seg_x(2*i)-1);
       project_y = sum(temp,2);
       seg_y = findEdge(project_y);
       temp = temp(seg_y(1):seg_y(2),:);
       temp = makeRec(temp);
       result(:,:,i) = imresize(temp, [50 50]);
    end
    result = addedge(result);
%     result(result~=0)=1;
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

function seg = findEdge(project_x) 
    %find each number's start point and end point
    %blank means wish to find blank (project == 0)
    seg = 0;
    num = 1;
    blank = false;
    for i = 1:length(project_x)
        if (project_x(i) == 0) == blank
            seg(num) = i;
            num = num + 1;
            blank = ~blank;
        end
            
    end
end

function newImgs = addedge(imgs)
    % add an 50% margin to each side of the image
    zeroline = zeros(15, 50);
    zerocolumn = zeros(80, 15);
    newImgs = zeros(28, 28, size(imgs,3));
    for i = 1:size(imgs, 3)
        temp = [zeroline;imgs(:,:,i);zeroline];
        temp = abs(imresize([zerocolumn temp zerocolumn], [28 28], 'bilinear'));
        %normalize to match the training data
        newImgs(:,:,i) = sqrt(temp/(max(temp(:)) - min(temp(:))));
    end
end
     
     
     
     
     
     
     
     
    