clear all;
clc

       in=input('Enter any number from 1 to 10 to check for results \n');
      
       var=strcat('FinalDataset\input',num2str(in));
       var = strcat(var,'.jpg');
       result=main(var);


