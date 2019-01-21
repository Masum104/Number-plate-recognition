function [ output ] = transform( input )

Filled=imfill(input,'holes');
% figure
% imshow(Filled);
% title('Before erode');
output=bwmorph(Filled,'thin',1);

 
output=imerode(output,strel('line',3,90));
% figure
% imshow(output);
% title('After erod');

end

