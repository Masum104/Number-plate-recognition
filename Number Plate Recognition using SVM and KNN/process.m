function [ output ] = process(I)

%ignores top 1/3
%algo to take out plate

[r,c]=size(I);
 

%x1=1;
%x2=c;
%y1=floor((1/3)*r);
%y2=r;
%I=imcrop(I, [x1 y1 x2-x1 y2-y1]);
%figure();
%imshow(I);
%Inew=imcrop(I,[1,maxStart,c-1,maxEnd-maxStart]);

x1=1;
x2=c;
y1=floor((1/3)*r);
y2=r;

for i=1:r
    for j=1:c
        if(j>=x1 && j<=x2 && i>=y1 && i<=y2)
            ;
        else
            I(i,j)=0;
        end
    end
end


[r,c]=size(I);
disp(r);
disp(c);
h=ones(5,5)/25;
I=imfilter(I,h);
% level = graythresh(I)
%
% I=im2bw(I,level);
for i1=1:r
    for i2=1:c
        if(I(i1,i2)<0.5)
            I(i1,i2)=0;
        else
            I(i1,i2)=1;
        end
    end
end

 
rowSum=zeros(r,1);
s=0;
for i=1:r
    sum=0;
    for j=1:c-1
        if(I(i,j)~=I(i,j+1))
            sum=sum+1;
        end
    end
    rowSum(i,1)=sum;
    s=s+sum;
end
disp(rowSum);
%avg=s/r;


start=1;
last=1;

maxLen=1;
maxStart=1;
maxEnd=1;
min_ele=20;
max_ele=48;

for i=2:r
    if(rowSum(i,1)>min_ele && rowSum(i,1)<max_ele)
        last=last+1;
    else
        diff=last-start;
        if(diff>=maxLen)
            maxLen=diff;
            maxStart=start;
            maxEnd=last;
        end
        start=i;
        last=i;
    end
end

maxStart
maxEnd
maxLen


if(maxStart>=20)
    maxStart=maxStart-20;
end

if(maxEnd<=r-20)
    maxEnd=maxEnd+20;
end


x1=1;
x2=c;
y1=maxStart;
y2=maxEnd;


for i=1:r
    for j=1:c
        if(j>=x1 && j<=x2 && i>=y1 && i<=y2)
            ;
        else
            I(i,j)=0;
        end
    end
end


%Inew=imcrop(I,[1,maxStart,c-1,maxEnd-maxStart]);
 output=I;
end
