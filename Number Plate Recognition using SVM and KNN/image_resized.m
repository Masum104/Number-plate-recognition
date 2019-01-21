
function image= image_resized(A)

A=A>graythresh(A)*255;
[row1,col1]=size(A);

neighbourhood=8;
Labelled=bwlabel(A,neighbourhood);

regions=regionprops(Labelled,'BoundingBox');
var=[];%cell(1,1);

%one connected component
reg=regions(1);
Array=[];
for elem =reg.BoundingBox
Array=[Array ceil(elem)];
end
myReg=Array;
 
  width=myReg(3);
  height=myReg(4);
  
  x_comp=[myReg(1)-2 myReg(1)+width+2];
  y_comp=[myReg(2)-2 myReg(2)+height+2];
  if x_comp(1)<1 || y_comp(1)<1
      if x_comp(1)<1
         x_comp(1)=1;
      else    
       y_comp(1)=1;
      end
  end
  if x_comp(2)>col1 || y_comp(2)>row1
     if(x_comp(2)>col1)
         x_comp(2)=col1;
     else
          y_comp(2)=row1;

     end
     
  end
  
  A=(Labelled==1);
  value=A(y_comp(1):y_comp(2),x_comp(1):x_comp(2));
   
  B = imresize(value,[20,20]);
  
image=B;
end