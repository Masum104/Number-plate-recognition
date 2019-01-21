function [ output ] = generateMatrix(input,value )

[sizex,sizey]=size(input);
result=zeros(sizex,sizey);

[row,col] = find(input==value);
conn = [row col];
[row1 col1]=size(conn);

for i=1:row1
    x1=conn(i,1);
    y1=conn(i,2);
    result(x1,y1)=255;
end

output=result;

end

