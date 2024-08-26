clear;

dataset_name = 'Yahoo';
dataset_path = ['E:\test\data\',dataset_name];
load([dataset_path,'\test.mat']);
Test = test/5.0;
K = 10;
alpha = 0.001;
beta = 0.001;

binary_code_bits = [8 16 32 64 128 256];
for r = binary_code_bits
    
    load([dataset_name,'\B_r=',num2str(r),'.mat'],'B');
    load([dataset_name,'\D_r=',num2str(r),'.mat'],'D');

    % test
    [old_ndcg,rmse,mae] = rating_metric(Test, B', D', K);
    [hit_sp,mrr_sp,ndcg] = rank_metric(K, B, D, Test);
    AUC = AUC_function(B,D,Test);

    % save result
    fid=fopen([dataset_name,'\rating45=P_rating=123N_ranking_result.txt'],'a');
        fprintf(fid,'binary_code_bit = %d; alpha = %f; beta = %f; K = %d \n',r,alpha,beta,K);
        fprintf(fid,'MAE = %f \t RMSE = %f \t MRR = %f \t AUC = %f\n',mae,rmse,mrr_sp(K),AUC);
        fprintf(fid,'New_NDCG@K = ');
        for i=1:K
            fprintf(fid,'%f ',ndcg(i));
        end
        fprintf(fid,'\n');
        fprintf(fid,'Old_NDCG@K = ');
        for i=1:K
            fprintf(fid,'%f ',old_ndcg(i));
        end
        fprintf(fid,'\n');
        fprintf(fid,'Hit@K = ');
        for i=1:K
            fprintf(fid,'%f ',hit_sp(i));
        end
        fprintf(fid,'\n');
        fclose(fid);
end
