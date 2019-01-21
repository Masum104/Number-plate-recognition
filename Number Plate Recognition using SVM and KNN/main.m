function finl_result=main(im)
im=imread(im);
figure
imshow(im);
title('Input Image');
im1=im2double(im);
im1=rgb2gray(im1);
figure
imshow(im1);
title('Gray Scale Image');
tempSize=size(im1);

im1=process(im1);
figure
imshow(im1);
title('Preprocessed Image');

folderName='dataset';
files=dir(folderName);
disp(length(files));
for k = 1 : length(files)
     baseFileName = files(k).name;
     fullFileName = fullfile(folderName, baseFileName);
     delete(fullFileName);
end

folderName='correct';
files=dir(folderName);
disp(length(files));
for k = 1 : length(files)
     baseFileName = files(k).name;
     fullFileName = fullfile(folderName, baseFileName);
     delete(fullFileName);
end



im1=medfilt2(im1,[3 3]);

B = edge(im1,'canny');

 figure
imshow(B);
 title('Edge Detection');
%% Contrast  Enhancements
temp=ones(3,3);
temp=padarray(temp,[1,1],0,'both');


B=conv2(double(B),double(temp));
% figure
% imshow(B);
% title('After');
% 
% %figure
%imshow(B);
%% Horizontal line removal
[sizex,sizey]=size(B);
% figure
% imshow(B);
% title('Before');

er=imerode(B,strel('line',100,0));
out1=imsubtract(B,er);
% figure
% imshow(out1);
% title('After');
%% filling of holes and eroding
output=transform(out1);
figure
imshow(output);
title('Holes filling and line Removed Image');


%% Horizontal line eroding
 output=imerode(output,strel('line',2,0));
 
%% finding connected components
B = bwlabel(output,8);
tempArr=max(B);
[row1,col1]=size(tempArr);

numConnComponents=0;

for i=1:row1
    for j=1:col1
        if(tempArr(i,j)>numConnComponents)
            numConnComponents=tempArr(i,j);
        end
    end
end

%% Extract all the connected components and put into a folder 'correct'
for var=1:numConnComponents
    
    [output]=generateMatrix(B,var);
    count=0;
    
    baseFileName = sprintf('%d.jpg', var);
    path='C:\Users\Masum\Documents\NumberPlateDetection-master\NumberPlateDetection-master\correct';
    fullFileName = fullfile(path, baseFileName);
    imwrite(output, fullFileName);
    
end
%% Applying threshold to remove irrelevant images and putting right images to 'dataset' folder
connected_images = FindImages('./correct/', true);

 counter=0;



for i = 1 : length(connected_images)
    %file name
    counter=counter+1;
    imgFile = char(connected_images(i));
    img = imread(imgFile);
    imshow(img);
    [row,col]=size(img);
    count=0;
    for k=1:row
        for l=1:col
            if(img(k,l)==1)
                count=count+1;
            end
        end
    end
    disp(count);
    if(count>10)
        length_of_file_name=length(imgFile);
        new_name_trimmed = imgFile(11:length_of_file_name-4);
        
        baseFileName = sprintf('%s.jpg', new_name_trimmed); 
        path='C:\Users\Masum\Documents\NumberPlateDetection-master\NumberPlateDetection-master\dataset';
        fullFileName = fullfile(path, baseFileName); 
        figure
        imshow(img);
        final_resized=image_resized(img);
        uint8Image = uint8(255 * final_resized);
        
        imwrite(uint8Image, fullFileName);
    end
 
    
    disp(imgFile);
end
finl_result=[];
end