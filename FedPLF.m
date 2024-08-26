function [W, Q, rmse_min,mae_min,hit_max,mrr_max,ndcg_max] = FedPLF(maxS, minS, S, Test, r, alpha, beta, dir, option)

[m,n] = size(S);

converge = false;
K = 10;

if isfield(option,'maxItr')
    R = option.R;
else
    R = 1000;
end
if isfield(option,'local_maxItr')
    local_maxItr = option.local_maxItr;
else
    local_maxItr = 1;
end

U = rand(r, m)*2 - 1;
V = rand(r, n)*2 - 1;
W0 = sign(U); W0(W0 == 0) = -1;
Q0 = sign(V); Q0(Q0 == 0) = -1;

W = W0;
Q = Q0;

ST = S';
S0 = S;
S0T = S';
IDX = (S0~=0);
IDXT = IDX';
nIDX = (S0 == 0);
nIDXT = (S0T == 0);


[loss,obj] = FedPLFobj(maxS,minS,S,IDX,W,Q,alpha,beta);
disp('Starting DCF...');

disp(['loss value = ',num2str(loss)]);
disp(['obj value = ',num2str(obj)]);

rmse_min = Inf;
mae_min = Inf;
ndcg_max = 0;
hit_max = 0;
mrr_max = 0;
delay = 0;

t = 0;

secret = false;
max_ratio = 0.3;
min_ratio = 0.2;
p_max = round(max_ratio * m);
p_min = round(min_ratio * m);

% Train
while ~converge
    
    t = t + 1;

    fid=fopen([dir, '\train_result_test.txt'],'a');
    fprintf(fid,'binary_code_bit = %d; alpha = %f; beta = %f; K = %d; \n',r,alpha,beta,K);
    
    delta_Qk = zeros(n, m);
    for k = 1:r
        delta_Qk = delta_Qk.*0;
        for u = 1:m
            delta_Quk = zeros(n, 1);
            % local_update according to I_u
            wu = W(:,u);
            Su = S0T(:,u);
            rated_items = find(IDXT(:,u));
            Qu = Q(:,rated_items);
            [delta_Quk_Iu, delta_wuk_Iu] = local_delta_compute(wu, Qu, ScaleScore(nonzeros(Su),r,maxS,minS), k);
            W(:,u) = local_wu_update(k, wu, delta_wuk_Iu, alpha);
            % if secret open send the part if delta_Quk to other clients
            delta_Quk(rated_items) = sign(delta_Quk_Iu);
            if secret == true
                p = randi([p_min, p_max], 1);
                p_client_ids = randi([1, m], p, 1);
                parts_delta_Quk = zeros(n, p+1);
                while sum(parts_delta_Quk, 2) ~= delta_Quk
                    parts_delta_Quk(:,u) = sign(rand(n, p+1)*2 - 1);
                end
                % send parts_delta_Quk to other client, keep one in local.
                delta_Qk(:, [p_client_ids; u]) = delta_Qk(:, [p_client_ids; u]) + parts_delta_Quk;
            else
                delta_Qk(:,u) = delta_Quk;
            end             
        end
        % all client send final_delta_Quk to server.
        Q(k,:) = global_Qk_update(k, Q, delta_Qk, beta);
    end

    
    % Test
    [loss,obj] = FedPLFobj(maxS, minS, S0, IDX, W, Q, alpha, beta);
    disp(['loss value = ',num2str(loss)]);
    disp(['obj value = ',num2str(obj)]);
    disp(['DCF at bit ',int2str(t),' Iteration:',int2str(t)]);
    
    [rmse,mae] = rating_loss(Test, W', Q');
    [hit_sp,mrr_sp,ndcg] = rank_metric(10, W, Q, Test);
    
    fprintf('round %d : RMSE = %f MAE = %f NDCG = %f HIT = %f MRR = %f\n',t,rmse,mae,ndcg(10),hit_sp(10),mrr_sp(10));
    fprintf(fid,'round %d : RMSE = %f MAE = %f NDCG = %f HIT = %f MRR = %f\n',t,rmse,mae,ndcg(10),hit_sp(10),mrr_sp(10));
    fclose(fid);
    if (mrr_sp(10) > mrr_max)
        mrr_max = mrr_sp(10);
        delay = 0;
    else
        delay = delay + 1;
    end
    if (ndcg(10) > ndcg_max)
        ndcg_max = ndcg(10);
        delay = 0;
    else
        delay = delay + 1;
    end
    
    if (hit_sp(10) > hit_max)
        hit_max = hit_sp(10);
        delay = 0;
    else
        delay = delay + 1;
    end
    if (mae < mae_min)
        mae_min = mae;
        delay = 0;
    else
        delay = delay + 1;
    end
    if (rmse < rmse_min)
        rmse_min = rmse;
        delay = 0;
    else
        delay = delay + 1;
    end
    
    if delay > 20
        break;
    end

    if t >= R || (sum(sum(W~=W0)) == 0 && sum(sum(Q~=Q0)) == 0)
        converge = true;
    end
end

end

function [loss,obj] = FedPLFobj(maxS,minS,S,IDX,W,Q,alpha,beta)
[~,n] = size(S);
r = size(W,1);
loss = zeros(1,n);
W(W == -1) = 0;
Q(Q == -1) = 0;
parfor j = 1:n
    dj = Q(:,j); 
    Wj = W(:,IDX(:,j)); 
    pred = (1 - sum(Wj&(~dj),1)/r)*4*r - 3*r;
    Sj = ScaleScore(nonzeros(S(:,j)),r,maxS,minS);
    loss(j) = sum((Sj' - pred).^2);
end
loss = sum(loss); 
W(W == 0) = -1;
Q(Q == 0) = -1;
obj = loss + alpha*sum(W,'all') + beta*sum(Q,'all');
end

function [delta_Quk, delta_wuk] = local_delta_compute(wu, Qu, Su, k)
wu_notk = sum(wu) - wu(k);
Qu_notk = (sum(Qu, 1) - Qu(k, :))';
wuQu_notk = Qu'*wu - Qu(k, :)'*wu(k);
err_hat_Su =  Su -(wuQu_notk - wu_notk + Qu_notk - 1);
delta_Quk = err_hat_Su *(wu(k) + 1);
delta_wuk = err_hat_Su .*(Qu(k, :) - 1)';
end

function wu = local_wu_update(k, wu, delta_wuk, alpha)
wu_notk = sum(wu) - wu(k);
wuk_hat = -sign(sum(delta_wuk) - alpha*wu_notk); % -sign(sum(delta_wuk - (alpha*wu_notk)/length(delta_wuk));
if wuk_hat ~= 0
    wu(k) = -wuk_hat;
end
end

function Qk = global_Qk_update(k, Q, delta_Qk, beta)
Q_notk_sum = sum(Q,1) - Q(k, :);
Qk = sign(sum(delta_Qk,2)' - beta * Q_notk_sum);
end