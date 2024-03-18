clc; clear;
addpath('C:\Users\kyh99\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\vlfeat-0.9.21\toolbox\mex\mexw64');
addpath('C:\Users\kyh99\Desktop\sfm\');

result = extractMatcher('0001.jpg', '0002.jpg');
[forFindRt_1, forFindRt_2, bestE, bestInliers,inf] = ransac(result.Q1, result.Q2, 0.000179);
[R1, t1, R2, t2] = Emat2Rt(bestE);
[R,t] = chooseRt(forFindRt_1, forFindRt_2, R1, t1, R2, t2);
fprintf('best R,t =\n');disp(R);disp(t);
fprintf('R1:\n');disp(R1);
fprintf('t1:\n');disp(t1);
fprintf('R2:\n');disp(R2);
fprintf('t2:\n');disp(t2);

% Inliers 끼리 matching 확인차원으로------------------------------
img1 = imread('0001.jpg');
img2 = imread('0002.jpg');
img1 = rgb2gray(img1);
img2 = rgb2gray(img2);

if size(img1, 1) ~= size(img2, 1)
    img2 = padarray(img2, [abs(size(img1, 1) - size(img2, 1)), 0], 0, 'post');
end

img_combined = [img1 img2]; 
figure;
imshow(img_combined);
hold on;
offset = size(img1, 2);

for i = 1:length(bestInliers)
    point1 = result.Q1(1:2, bestInliers(i)); 
    point2 = result.Q2(1:2, bestInliers(i)); 
    point2(1) = point2(1) + offset; 
    line([point1(1), point2(1)], [point1(2), point2(2)], 'Color', 'r', 'LineWidth', 1.5);
end
% fprintf('%d\n',bestInliers);
hold off;

function [R, t] = chooseRt(forFindRt_1, forFindRt_2, R1, t1, R2, t2)

    dis_1 = 0;
    dis_2 = 0;
    dis_3 = 0;
    dis_4 = 0;
    
    for i = 1:size(forFindRt_1, 2)
        tr_1 = R1 * forFindRt_1(:, i) + t1;
        tr_2 = R1 * forFindRt_1(:, i) + t2;
        tr_3 = R2 * forFindRt_1(:, i) + t1;
        tr_4 = R2 * forFindRt_1(:, i) + t2;
        
        dis_1 = dis_1 + norm(tr_1 - forFindRt_2(:, i));
        dis_2 = dis_2 + norm(tr_2 - forFindRt_2(:, i));
        dis_3 = dis_3 + norm(tr_3 - forFindRt_2(:, i));
        dis_4 = dis_4 + norm(tr_4 - forFindRt_2(:, i));
    end
    [M, I] = min([dis_1, dis_2, dis_3, dis_4]);
    fprintf('min array = \n');disp([dis_1, dis_2, dis_3, dis_4]);

    if I == 1
        R = R1; t = t1;
    elseif I == 2
        R = R1; t = t2;
    elseif I == 3
        R = R2; t = t1;
    else
        R = R2; t = t2;
    end
end

function [R1, t1, R2, t2] = Emat2Rt(bestE)
    [U, ~, V] = svd(bestE);
    % D = diag([1, 1, 0]);
    W = [0 -1 0; 1 0 0; 0 0 0]; 
    t = U(:, end);

    R1 = U*W*V';
    R2 = U*W'*V';
    t1 = U(:,3);
    t2 = -U(:,3);

    t1 = t;
    t2 = -t;
end

function result = extractMatcher(img1Path, img2Path)
    % read and col2gray
    img1 = imread(img1Path);
    img2 = imread(img2Path);
    img1 = single(rgb2gray(img1));
    img2 = single(rgb2gray(img2));

    % vl_sift를 사용해서 feature 추출
    [f1, d1] = vl_sift(img1);
    [f2, d2] = vl_sift(img2);
    % f1은 특징점의 위치, 스케일, 방향 정보를 담고 있는 4xN 행렬
    % d1은 각 특징점에 대한 descriptor, 128xN 행렬

    fprintf('img1 feature 개수: %d\n', size(f1, 2));
    fprintf('img2 feature 개수: %d\n', size(f2, 2));

    % -----------길이 비가 a인 것을 만족하는 matches 출력-----------
    a = 1.5;
    [matches, score] = vl_ubcmatch(d1, d2, a);
    fprintf('matches 개수: %d\n', size(matches,2));
    matchedPoints1 = f1(1:2, matches(1, :)); % img1의 매칭된 특징점 위치
    matchedPoints2 = f2(1:2, matches(2, :)); % img2의 매칭된 특징점 위치
    
    Q1 = [matchedPoints1; ones(1, size(matchedPoints1, 2))];
    Q2 = [matchedPoints2; ones(1, size(matchedPoints2, 2))];

    % fprintf('q1 size: %d\n',size(Q1, 1));
    % fprintf('value of Q2[3,1]: %d\n', Q2(3,1));
    % ----------------------------plot%----------------------------
    % subplot(1, 2, 1);
    % imshow(uint8(img1));
    % hold on;
    % plot(f1(1, matches(1, :)), f1(2, matches(1, :)), 'b*');
    % 
    % subplot(1, 2, 2);
    % imshow(uint8(img2));
    % hold on;
    % plot(f2(1, matches(2, :)), f2(2, matches(2, :)), 'r*');
    % ---------------img에서 score 제일 높은 놈 찾기---------------
    % [maxScore, maxIndex] = max(score); 
    % bestMatchIndex = matches(:, maxIndex);
    % bestMatchFeature1 = f1(:, bestMatchIndex(1));
    % bestMatchFeature2 = f2(:, bestMatchIndex(2));
    % fprintf('가장 높은 스코어: %f\n', maxScore);
    % fprintf('img1의 pixel좌표: (%f, %f)\n', bestMatchFeature1(1), bestMatchFeature1(2));
    % fprintf('img2의 pixel좌표: (%f, %f)\n', bestMatchFeature2(1), bestMatchFeature2(2));

    result.Q1 = Q1;
    result.Q2 = Q2;
end

function [forFindRt_1, forFindRt_2, bestE, bestInliers, minError] = ransac(Q1, Q2, threshold)
p = 0.99;
iter = log(1-p)/log(1/9); % N=Log(1-p)/Log(1-(1-e)^s)
minError = inf;
K =[1698.873755 0.000000     971.7497705
0.000000    1698.8796645 647.7488275
0.000000    0.000000     1.000000]; %intrinsic Parameter
K_inv = inv(K); % (=k.inv())
K_inv_T = K_inv'; % (=k.inv().transpose())
    bestNumInliers = 0;
    bestE = [];
    bestInliers = [];
    forHistogram = [];

    forFindRt_1 = [];
    forFindRt_2 = [];
    for i = 1:iter
        % 일단 951개(0318기준)중에 같은 열 5개 랜덤 샘플링
        indices = randperm(size(Q1,2), 5);
        sampleQ1 = Q1(:, indices);
        sampleQ2 = Q2(:, indices);

        % E matrix 계산
        Evec = calibrated_fivepoint(sampleQ1, sampleQ2);

        % 각 에센셜 매트릭스에 대해 인라이어수 계산
        for j = 1:size(Evec, 2) % E matrix에서 복소수 제거하고 몇갠지 랜덤이라
            E = reshape(Evec(:,j),3,3); %E matrix 3x3
            % inliers = [];
            currentInliers = [];
            curForFindRt_1 = []; %Inlier시에 img1 pixel좌표 담을 배열
            curForFindRt_2 = []; %Inlier시에 img2 pixel좌표 담을 배열
            
            currentError = 0;
            for k = 1:size(Q1, 2)
                error = abs(Q2(:,k)'*K_inv_T * E*K_inv * Q1(:,k)); %최대한 0에 가깝게 나오길 바라며.. 
                currentError = currentError + error;
                % forHistogram = [forHistogram, error]; % iter 많이하고 histogram을 그려서 threshold 값 설정
                if error < threshold
                    % inliers = [inliers, k];
                    curForFindRt_1 = [curForFindRt_1, Q1(:, k)]; % Add the inlier from img1
                    curForFindRt_2 = [curForFindRt_2, Q2(:, k)]; % Add the inlier from img2
                    currentInliers = [currentInliers, k];
                end
            end
            if length(currentInliers) > bestNumInliers ||(length(currentInliers) == bestNumInliers && currentError < minError)
                bestNumInliers = length(currentInliers);
                bestE = E;
                bestInliers = currentInliers;
                forFindRt_1 = curForFindRt_1;
                forFindRt_2 = curForFindRt_2
                minError = currentError; % 극한의 Inlier 찾기
            end
        end
    end

% figure; 
% histogram(forHistogram, 50);
    fprintf('최소 에러: %f\n', minError);
    fprintf('인라이어 수: %d\n', bestNumInliers);
end
