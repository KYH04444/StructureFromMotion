clc; clear;
addpath('C:\Users\kyh99\AppData\Roaming\MathWorks\MATLAB Add-Ons\Collections\vlfeat-0.9.21\toolbox\mex\mexw64');
addpath('C:\Users\kyh99\Desktop\sfm\');

result = extractMatcher('0001.jpg', '0002.jpg');
[forFindRt_1, forFindRt_2, bestE, bestInlier_idx,inf] = ransac(result.Q1, result.Q2, 0.000179);
[R1, t1, R2, t2] = Emat2Rt(bestE);

fprintf('R1:\n');disp(R1);
fprintf('t1:\n');disp(t1);
fprintf('R2:\n');disp(R2);
fprintf('t2:\n');disp(t2);
% fprintf('R:\n');disp(R);
% fprintf('t:\n');disp(t);
Data = [];

for i=1:3
    [beforePmatrix, curPmatrix, P_3D_1] = triangulation(R1, t1, forFindRt_1(1:2,i), forFindRt_2(1:2,i), true);
    [~, ~, P_3D_2] = triangulation(R1, t2, forFindRt_1(1:2,i), forFindRt_2(1:2,i), true);
    [~, ~, P_3D_3] = triangulation(R2, t1, forFindRt_1(1:2,i), forFindRt_2(1:2,i), true);
    [~, ~, P_3D_4] = triangulation(R2, t2, forFindRt_1(1:2,i), forFindRt_2(1:2,i), true);
        if (P_3D_1(3) > 0)
            P_3D = P_3D_1;
        elseif (P_3D_2( 3) > 0)
            P_3D = P_3D_2;
        elseif (P_3D_3( 3) > 0)
            P_3D = P_3D_3;
        elseif (P_3D_4( 3) > 0)
            P_3D = P_3D_4;
        end
    Data(i,1:6) = [forFindRt_1(1:2,i)',1, P_3D'];
    % fprintf("Point on 3D: \n ");disp(P_3D);
end

% disp(Data);
ProjectionMatrix=PerspectiveThreePoint(Data);
fprintf('ProjectionMatrix: \n');disp(ProjectionMatrix);
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

for i = 1:length(bestInlier_idx)
    point1 = result.Q1(1:2, bestInlier_idx(i)); 
    point2 = result.Q2(1:2, bestInlier_idx(i)); 
    point2(1) = point2(1) + offset; 
    line([point1(1), point2(1)], [point1(2), point2(2)], 'Color', 'r', 'LineWidth', 1.5);
end
hold off;
% Inliers 끼리 matching 확인차원으로------------------------------

function [beforePmatrix, curPmatrix, P_3D] = triangulation(R, t, point2D_1, point2D_2, init)

    K =[1698.873755 0.000000 971.7497705;
        0.000000 1698.8796645 647.7488275;
        0.000000 0.000000 1.000000]; 
    beforePmatrix= [];
    if init == true
        R_init = eye(3);
        t_init = zeros(3,1);
        P_init = K * [R_init, t_init];
        beforePmatrix = P_init;    
    end

        curPmatrix = K*[R,t];
        
        x_1 = point2D_1(1, 1);
        y_1 = point2D_1(2, 1);

        x_2 = point2D_2(1, 1);
        y_2 = point2D_2(2, 1);

        curP1 = curPmatrix(1,:);
        curP2 = curPmatrix(2,:);
        curP3 = curPmatrix(3,:);
        
        beforeP1 = beforePmatrix(1,:);
        beforeP2 = beforePmatrix(2,:);
        beforeP3 = beforePmatrix(3,:);

        A = [x_1*curP3 - curP1;
             y_1*curP3 - curP2;
             x_2*beforeP3 - beforeP1;
             y_2*beforeP3 - beforeP2;];

    [~, ~, V] = svd(A);
    P_3D = V(:, end);
    P_3D = P_3D ./ P_3D(4); % 동차 좌표를 일반 좌표로 변환
    
    P_3D = P_3D(1:3); % 마지막 요소(동차 좌표의 스케일) 제거
    beforePmatrix = curPmatrix;
end


function [R1, t1, R2, t2] = Emat2Rt(bestE)
    % D = diag([1, 1, 0]);
    [U, ~, V] = svd(bestE);
    
    W = [0 -1 0; 1 0 0; 0 0 0]; 
    R1 = U*W*V';
    R2 = U*W'*V';
    t1 = U(:,3);
    t2 = -U(:,3);
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
    result.Q1 = Q1;
    result.Q2 = Q2;
end

function [forFindRt_1, forFindRt_2, bestE, bestInlier_idx, minError] = ransac(Q1, Q2, threshold)
p = 0.99;
iter = log(1-p)/log(1-(1-0.5)^5); % N=Log(1-p)/Log(1-(1-e)^s)
minError = inf;
K =[1698.873755 0.000000     971.7497705;
0.000000    1698.8796645 647.7488275;
0.000000    0.000000     1.000000]; %intrinsic Parameter
K_inv = inv(K); % (=k.inv())
K_inv_T = K_inv'; % (=k.inv().transpose())

    bestNumInliers = 3;
    bestE = [];
    bestInlier_idx = [];
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
                    curForFindRt_1 = [curForFindRt_1, Q1(:, k)]; 
                    curForFindRt_2 = [curForFindRt_2, Q2(:, k)];
                    currentInliers = [currentInliers, k];
                end
            end
            if (length(currentInliers) == bestNumInliers && currentError < minError)
                bestNumInliers = length(currentInliers);
                bestE = E;
                bestInlier_idx = currentInliers;
                forFindRt_1 = curForFindRt_1;
                forFindRt_2 = curForFindRt_2;
                minError = currentError; % 극한의 Inlier 찾기
            end
        end
    end

% figure; 
% histogram(forHistogram, 50);
    fprintf('최소 에러: %f\n', minError);
    fprintf('인라이어 수: %d\n', bestNumInliers);
end
