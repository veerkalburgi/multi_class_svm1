clc;
clear;
close all;
%% import data
a=csvread('dataset.csv');
Y=a(:,1);
X=a(:,2:end);
 %%
 
 %Train an ECOC multiclass model
 
 Mdl = fitcecoc(X,Y);
 
 %Display the coding design matrix
 
 Mdl.ClassNames;
 CodingMat = Mdl.CodingMatrix;
 
 %Binary learner 
 
 Mdl.BinaryLearners{1};
 Mdl.BinaryLearners{2};
 Mdl.BinaryLearners{3};

 %In-sample classification error
 
 isLoss = resubLoss(Mdl)

 %Cross-Validate ECOC Classifier
 
 t = templateSVM('Standardize',1);
 
 %Train the ECOC classifier

 Mdl = fitcecoc(X,Y,'Learners',t,...
    'ClassNames',{'1','2','3'});

%Cross-validate MDL

CVMdl = crossval(Mdl);

%Generalization error

oosLoss = kfoldLoss(CVMdl);

%Posterior Probabilities Using ECOC Classifiers

t = templateSVM('Standardize',1,'KernelFunction','gaussian');
Mdl = fitcecoc(X,Y,'Learners',t,'FitPosterior',1,...
    'ClassNames',{'1','2','3'},...
    'Verbose',2);

%In-sample labels and class posterior probabilities

[label,~,~,Posterior] = resubPredict(Mdl,'Verbose',1);
Mdl.BinaryLoss

%Display a random set of results.

idx = randsample(size(X,1),20,1);
Mdl.ClassNames
table(Y(idx),label(idx),Posterior(idx,:),...
    'VariableNames',{'TrueLabel','PredLabel','Posterior'})
%%



