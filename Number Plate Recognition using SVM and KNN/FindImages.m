function [list] = FindImages(direc, Path)
% it gives  a list of the image files in the specified
%    directory.
%

data = dir(direc);
Index = [data.isdir];

% Get file list.
Initial_List = {data(~Index).name};

list = {};

for i = 1 : length(Initial_List)
    
    img = char(Initial_List(i));
    
    if (Path == true)
        img = strcat(direc, img);
    end
    
    list{i} = img;
end
end
