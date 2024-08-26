%all_dataset = {'Ciao','Epinion','Hetrec-ML','Ml1M','Yelp','Ml25M'}; %dataset_name = dataset_name{1};
all_dataset = {'Epinion'}; %dataset_name = dataset_name{1};
for dataset = all_dataset
    dataset_name = dataset{1};
    dir =['.\', dataset_name];
    dataset_path = ['.\dataset\',dataset_name];
    load([dataset_path,'\test.mat']);
    load([dataset_path,'\train.mat']);

    %target bit size
    binary_bit = [64];
    K = 10;
    paraments = [0.01];
    option.debug = true;

    %number of iterations
    option.R = 500;
    option.local_maxItr = 1;

    S = train;
    Test = test;
            
    [rows, cols] = size(S);
    [maxS,~] = max((S(:)));
    [minS,~] = min((S(:)));
   
    for r = binary_bit
        for alpha = paraments
            for beta = paraments
                % apply initialization
                rmse_min = Inf;
                mae_min = Inf;
                ndcg_max = 0;
                hit_max = 0;
                mrr_max = 0;
                [W,Q,rmse,mae,hit,mrr,ndcg] = FedPLF(maxS,minS,S,Test, r, alpha, beta, dir, option);
                if (mrr > mrr_max)
                    mrr_max = mrr;
                end
                if (ndcg > ndcg_max)
                    ndcg_max = ndcg;
                end
                if (hit > hit_max)
                    hit_max = hit;
                end
                if (mae < mae_min)
                    mae_min = mae;
                end
                if (rmse < rmse_min)
                    rmse_min = rmse;
                end
                fid=fopen([dir,'\rating_result_new_test.txt'],'a');
                fprintf(fid,'binary_code_bit = %d; alpha = %f; beta = %f; K = %d; \n',r,alpha,beta,K);
                fprintf(fid,'RMSE = %f \t MAE = %f \t  Hit = %f \t MRR = %f \t NDCG = %f\n',rmse_min,mae_min,hit_max,mrr_max,ndcg_max);
                fclose(fid);
            end
        end
    end
end
