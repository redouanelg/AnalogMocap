clc
clear all
close all

%set(0,'defaultlinelinewidth',2)
%set(gca,'DefaultTextFontSize',25)

%% load amc files
% files = dir('Training/*.amc');
% Dtrain=[];
% for k = 1:length(files)
%     Dtrain{k} = amc_to_matrix(strcat('Training/',files(k).name));
% end
% 
 %% Walk catalog
% catalogWalk=[];
% lengthsequence=[];
%  for k=1:22
%      tempcatalog=[Dtrain{k}(1:end-1,:) Dtrain{k}(2:end,:)];
%      lengthsequence=[lengthsequence size(tempcatalog,1)];
%      catalogWalk=[catalogWalk;tempcatalog];
%  end

%% Catalog
%  catalog=catalogWalk;
%  save('cataloglag0.mat','catalog','lengthsequence');
load('cataloglag0.mat');
indicesOfSequences=[0 cumsum(lengthsequence)];
analogs=catalog(:,56:62);
succesors=catalog(:,62+56:62+62);
rmsExpect=[];
rmsMax=[];

%% Test data
R=0.1;

D = amc_to_matrix('35_34.amc');

testdata=D(:,56:62)';
g = @(x,t) x(:,t);
yo=testdata+sqrt(R).*randn(size(testdata));

%MaskInd=(find(rand(size(yo))>0.5));
MaskInd=(7*149+1):(7*299+1);
length(MaskInd)
%yo(33:62,100:end)=NaN;
yo(MaskInd)=NaN;

%% In case you use embedding lagged states
lag=2;
laggedsuccessors=[];
laggedanalogs=[];

for j=2:length(indicesOfSequences)
tmpa=[];
tmps=[];
for i=1:lag+1
tmpa=[tmpa analogs(indicesOfSequences(j-1)+lag+2-i:indicesOfSequences(j)-i+1,:)];
tmps=[tmps succesors(indicesOfSequences(j-1)+lag+2-i:indicesOfSequences(j)-i+1,:)];
end
laggedanalogs=[laggedanalogs; tmpa];
laggedsuccessors=[laggedsuccessors; tmps];
end
analogsL=laggedanalogs;
succesorsL=laggedsuccessors;

%% Construction of Transition matrix

disp('Construction of Transition matrix..')
states=union(analogsL,succesorsL,'rows');
T_train=size(states,1);

scaledanalogs=(analogsL-repmat(mean(analogsL),size(analogsL,1),1))./repmat(std(analogsL),size(analogsL,1),1);
scaledstates=(states-repmat(mean(states),size(states,1),1))./repmat(std(states),size(states,1),1);


for k=8    
[index_wknn,dist_wknn]=knnsearch(scaledanalogs,scaledstates,'k',k,'NSMethod','kdtree');
%dist_wknn_norm=dist_wknn./median(dist_wknn(:)); %!!!!!!
dist_wknn_norm=(dist_wknn.^2)./(2.*repmat(var(dist_wknn')',1,k)); %!!!!!!
s=mk_stochastic(exp(-dist_wknn_norm));
%s(s<median(median(s)))=0;
[LIA,LOCB] = ismember(succesorsL(index_wknn(:),:),states,'rows');
transmat=sparse(repmat(linspace(1,T_train,T_train)',k,1),LOCB,s(:),T_train,T_train);
% 
%  figure
%  spy(transmat)
%  legend('Transition matrix non null elements')

nbpart=9000;
xt_train=states(:,1:size(testdata,1))';

%% Analog FB
prior=ones(T_train,1);
%[index_wknn_prior,dist_wknn_prior]=knnsearch(states,testdata(:,1)','k',k,'NSMethod','kdtree');
%prior(index_wknn_prior)=1;
[gamma, alpha] = Analog_Forward_Backward_mocap(prior, nbpart, yo, xt_train, R.*eye(size(testdata)),g,transmat);

ResFBexpect=xt_train*gamma;
U=ResFBexpect(MaskInd);
V=testdata(MaskInd);
rmsExpect=[rmsExpect rms(U(:)-V(:))]
ResFB=[];
for i=1:size(yo,2)
        [m1,ind]=max(gamma(:,i));
        ResFB=[ResFB xt_train(:,ind)];
end
UM=ResFB(MaskInd);
rmsMax=[rmsMax rms(UM(:)-V(:))]
end

figure
subplot(2,3,1)
plot(testdata(1,:)); hold on %righthand
%plot(yo(1,:),'go')
plot(ResFBexpect(1,:),'r--')
title('1','FontSize', 16);
subplot(2,3,2)
plot(testdata(2,:)); hold on %righthand
%plot(yo(2,:),'go')
plot(ResFBexpect(2,:),'r--')
title('2','FontSize', 16);
subplot(2,3,3)
plot(testdata(4,:)); hold on %righthand
%plot(yo(4,:),'go')
plot(ResFBexpect(4,:),'r--')
title('4','FontSize', 16);
subplot(2,3,4)
plot(testdata(5,:)); hold on %righthand
%plot(yo(5,:),'go')
plot(ResFBexpect(5,:),'r--')
title('5','FontSize', 16);
subplot(2,3,5)
plot(testdata(6,:)); hold on %righthand
%plot(yo(6,:),'go')
plot(ResFBexpect(6,:),'r--')
title('6','FontSize', 16);
subplot(2,3,6)
plot(testdata(7,:)); hold on %righthand
%plot(yo(7,:),'go')
plot(ResFBexpect(7,:),'r--')
title('7','FontSize', 16);
set(gcf,'PaperPositionMode', 'auto');
ll=legend('True','AnFB');
set(ll,'Position',[0.49495907738095 0.491564948764315 0.0440104166666667 0.0417985232067511])
print -depsc2 mocap2.eps
% 
%% naive nearest neighbors
InterpYo=zeros(size(yo));
for j=1:size(yo,2)
    j
    if sum(isnan(yo(:,j)))~=length(isnan(yo(:,j)))
        MM=knnimpute([yo(:,j) states'],k);
        InterpYo(:,j)=MM(:,j);
    else
        InterpYo(:,j)=nanmean(yo,2);
    end
end
UNNN=InterpYo(MaskInd);
rmsNNN=rms(UNNN(:)-V(:))

figure
plot(testdata(1,:)); hold on %righthand
plot(yo(1,:),'g*')
plot(InterpYo(1,:),'r')
%% Matrix to AMC
resultExpect=[D(:,1:55) ResFBexpect'];
obs=[D(:,1:55) yo'];
resultNNN=[D(:,1:55) InterpYo'];

matrix_to_amc('ResFBexpect.amc',resultExpect);
matrix_to_amc('ResNNN.amc',resultNNN);
matrix_to_amc('obs.amc',obs);

%% motion
skel = acclaimReadSkel('35.asf');
[channels, skel] = acclaimLoadChannels('35_34.amc', skel);
[channels2, skel] = acclaimLoadChannels('ResFBexpect.amc', skel);
[channelsobs, skel] = acclaimLoadChannels('obs.amc', skel);
[channelsNNN, skel] = acclaimLoadChannels('ResNNN.amc', skel);


skelPlayData(skel, channels, 1/120,'true.avi');
skelPlayData(skel, channels2, 1/120,'analogFB.avi');
skelPlayData(skel, channelsobs, 1/120,'observations.avi');
skelPlayData(skel, channelsNNN, 1/120,'naivenearestneighbors.avi');



figure
subplot(1,3,1)
imagesc(testdata)
title('True','FontSize', 14);
subplot(1,3,2)
bbb=imagesc(yo)
set(bbb,'AlphaData',~isnan(yo));
title('Obsevations','FontSize', 14);
subplot(1,3,3)
imagesc(ResFBexpect)
title('AnFB','FontSize', 14);
set(gcf,'PaperPositionMode', 'auto');
print -depsc2 obsNaN.eps

