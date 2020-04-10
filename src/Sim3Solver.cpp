/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Sim3Solver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>

#include "KeyFrame.h"
#include "ORBmatcher.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

namespace ORB_SLAM2
{

// 变量名称命名规则：m表示member，v表示vector(数据结构)，p表示point
// 待确认：vpMatched12是Frame2中MapPoints的匹配，大小为Frame2的特征点个数
// 里面存着匹配的Frame2的MapPoint指针或NULL，NULL表示未匹配
// 对应地，Frame1中的GetMapPointMatches函数可获得同样大小的vector，存着Frame1的MapPoint指针
// 注意：12表示2到1，这是视觉SLAM十四讲中就有的定义，后面的R和t也是一样
// MapPoint即为地图3D点，KeyPoint为投影后的2D图像关键点
// 本质上3D点存在于三维世界，同时被多个关键帧观测到，从而获得了不同帧内的地图点
// bFixScale为1表示Sim3，为0表示SE3
Sim3Solver::Sim3Solver(KeyFrame *pKF1, KeyFrame *pKF2, const vector<MapPoint *> &vpMatched12, const bool bFixScale) : mnIterations(0), mnBestInliers(0), mbFixScale(bFixScale)
{
    // mpKF1和mpKF2为头文件中定义的protected变量
    // 问题：使用protected的原因？可能是隔绝影响
    mpKF1 = pKF1;
    mpKF2 = pKF2;

    // 问题：GetMapPointMatches的定义是什么？需要研究下KeyFrame的构造
    vector<MapPoint *> vpKeyFrameMP1 = pKF1->GetMapPointMatches();

    mN1 = vpMatched12.size(); // mN1为pKF2特征点的个数

    mvpMapPoints1.reserve(mN1); // vector::reserve要求至少能容纳多少个元素
    mvpMapPoints2.reserve(mN1);
    mvpMatches12 = vpMatched12;
    mvnIndices1.reserve(mN1);
    mvX3Dc1.reserve(mN1);
    mvX3Dc2.reserve(mN1);

    cv::Mat Rcw1 = pKF1->GetRotation();
    cv::Mat tcw1 = pKF1->GetTranslation();
    cv::Mat Rcw2 = pKF2->GetRotation();
    cv::Mat tcw2 = pKF2->GetTranslation();

    mvAllIndices.reserve(mN1); // Indices for random selection

    size_t idx = 0;
    // mN1为pKF2特征点的个数
    for (int i1 = 0; i1 < mN1; i1++)
    {
        // 如果该特征点在pKF1中有匹配
        if (vpMatched12[i1])
        {
            // step1: 根据vpMatched12配对比配的MapPoint： pMP1和pMP2
            // 这里的逻辑基本清楚了，但还需要进一步确定
            MapPoint *pMP1 = vpKeyFrameMP1[i1];
            MapPoint *pMP2 = vpMatched12[i1]; //

            if (!pMP1)
                continue;

            if (pMP1->isBad() || pMP2->isBad())
                continue;

            // step2：计算允许的重投影误差阈值：mvnMaxError1和mvnMaxError2
            // 注：是相对当前位姿投影3D点得到的图像坐标，见step6
            // step2.1：根据匹配的MapPoint找到对应匹配特征点的索引：indexKF1和indexKF2
            int indexKF1 = pMP1->GetIndexInKeyFrame(pKF1);
            int indexKF2 = pMP2->GetIndexInKeyFrame(pKF2);

            if (indexKF1 < 0 || indexKF2 < 0)
                continue;

            // step2.2：取出匹配特征点的引用：kp1和kp2
            // 使用引用可以节约资源
            const cv::KeyPoint &kp1 = pKF1->mvKeysUn[indexKF1];
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[indexKF2];

            // step2.3：根据特征点的尺度计算对应的误差阈值：mvnMaxError1和mvnMaxError2
            // ORB特征使用了金字塔来实现尺度不变性，因此需要计算特征点当前的尺度
            // octave表示金字塔层数，即特征点被提取的层，OpenCV自带定义
            // 尺度因子的平方，其中尺度因子为scale^n，scale=1.2，n为层数
            // 问题：关于尺度的对应关系还需要研究
            const float sigmaSquare1 = pKF1->mvLevelSigma2[kp1.octave];
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];

            mvnMaxError1.push_back(9.210 * sigmaSquare1); // 问题：9.21是怎么来的？
            mvnMaxError2.push_back(9.210 * sigmaSquare2);

            // mvpMapPoints1和mvpMapPoints2是匹配的MapPoints容器
            mvpMapPoints1.push_back(pMP1);
            mvpMapPoints2.push_back(pMP2);
            mvnIndices1.push_back(i1);

            // step4：将MapPoint从世界坐标系变换到相机坐标系：mvX3Dc1和mvX3Dc2
            cv::Mat X3D1w = pMP1->GetWorldPos();
            mvX3Dc1.push_back(Rcw1 * X3D1w + tcw1); // 注意：cw表示从world到camera

            cv::Mat X3D2w = pMP2->GetWorldPos();
            mvX3Dc2.push_back(Rcw2 * X3D2w + tcw2);

            mvAllIndices.push_back(idx); // mvAllIndices给定了取点的范围
            idx++;
        }
    }

    // step5：两个关键帧的内参
    mK1 = pKF1->mK;
    mK2 = pKF2->mK;

    // step6：记录计算两帧Sim3之前3D mappoint在图像上的投影坐标：mvP1im1和mvP2im2
    FromCameraToImage(mvX3Dc1, mvP1im1, mK1);
    FromCameraToImage(mvX3Dc2, mvP2im2, mK2);

    SetRansacParameters();
}

// 设置RANSAC参数，按照特征点匹配数量动态调整参数，具体待研究
void Sim3Solver::SetRansacParameters(double probability, int minInliers, int maxIterations)
{
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;

    N = mvpMapPoints1.size(); // number of correspondences

    mvbInliersi.resize(N);

    // Adjust Parameters according to number of correspondences
    float epsilon = (float)mRansacMinInliers / N;

    // Set RANSAC iterations according to probability, epsilon, and max iterations
    int nIterations;

    if (mRansacMinInliers == N)
        nIterations = 1;
    else
        nIterations = ceil(log(1 - mRansacProb) / log(1 - pow(epsilon, 3)));

    mRansacMaxIts = max(1, min(nIterations, mRansacMaxIts));

    mnIterations = 0;
}

// Ransac求解mvX3Dc1和mvX3Dc2之间Sim3，函数返回mvX3Dc2到mvX3Dc1的Sim3变换
cv::Mat Sim3Solver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;
    vbInliers = vector<bool>(mN1, false);
    nInliers = 0;

    // 判断是否符合RANSAC返回条件
    if (N < mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat(); // 没有合适解
    }

    vector<size_t> vAvailableIndices;

    // 用cv::Mat的形式存3组匹配对比较方便
    cv::Mat P3Dc1i(3, 3, CV_32F);
    cv::Mat P3Dc2i(3, 3, CV_32F);

    int nCurrentIterations = 0;
    while (mnIterations < mRansacMaxIts && nCurrentIterations < nIterations)
    {
        // 说明有内外两层for循环
        // 问题：总的迭代次数没见到比当前的增加得快啊？
        nCurrentIterations++; // 这个函数中迭代的次数
        mnIterations++;       // 总的迭代次数，默认为最大为300

        vAvailableIndices = mvAllIndices; // 问题：每次迭代重置取点范围，那不就白剔除已经选过的了吗？

        // 以下为RANSAC的基本流程，共有4个步骤组成

        // Get min set of points
        // 步骤1：任意取三组点算Sim矩阵
        // 待确认：要求R,t,s，分别是3,2,1个自由度，因此3组点可以提供6个自由度的约束
        // 同时可以理解为求解R，最基本的方法就是3点法
        for (short i = 0; i < 3; ++i)
        {
            // 随机给定Index，并以此选择一对匹配对
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);

            int idx = vAvailableIndices[randi];

            // P3Dc1i和P3Dc2i中点的排列顺序：
            // x1 x2 x3 ...
            // y1 y2 y3 ...
            // z1 z2 z3 ...
            mvX3Dc1[idx].copyTo(P3Dc1i.col(i)); // 拷贝到P3Dc1i和P3Dc2i
            mvX3Dc2[idx].copyTo(P3Dc2i.col(i));

            // 已经选择过的需要删掉，以保证每次迭代中不会重复选择特征点
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }

        // 步骤2：根据两组匹配的3D点，计算之间的Sim3变换
        // 应该是三组吧？
        ComputeSim3(P3Dc1i, P3Dc2i);

        // 步骤3：通过投影误差进行inlier检测
        // 这里都不怎么需要传参好厉害，应该是和protected数据类型有关系
        CheckInliers();

        // 如果当前结果比之前的都要好，则更新最好的
        if (mnInliersi >= mnBestInliers)
        {
            mvbBestInliers = mvbInliersi;
            mnBestInliers = mnInliersi;
            mBestT12 = mT12i.clone();
            mBestRotation = mR12i.clone();
            mBestTranslation = mt12i.clone();
            mBestScale = ms12i;

            // 这一部分没理解在干嘛
            if (mnInliersi > mRansacMinInliers)
            {
                nInliers = mnInliersi;
                for (int i = 0; i < N; i++)
                    if (mvbInliersi[i])
                        vbInliers[mvnIndices1[i]] = true;

                // ！！！！！note: 1. 只要计算得到一次合格的Sim变换，就直接返回 2. 没有对所有的inlier进行一次refine操作
                return mBestT12;
            }
        }
    }

    // 超过最大迭代次数，返回无法求解
    if (mnIterations >= mRansacMaxIts)
        bNoMore = true;

    return cv::Mat();
}

// 里面就只有一个iterate函数，多加了一个Flag
cv::Mat Sim3Solver::find(vector<bool> &vbInliers12, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts, bFlag, vbInliers12, nInliers);
}

// ComputeSim3里要用，为了求得质心和去质心坐标
void Sim3Solver::ComputeCentroid(cv::Mat &P, cv::Mat &Pr, cv::Mat &C)
{
    // 这两句可以使用CV_REDUCE_AVG选项来搞定
    cv::reduce(P, C, 1, CV_REDUCE_SUM); // 矩阵P每一行求和
    C = C / P.cols;                     // 求平均

    for (int i = 0; i < P.cols; i++)
    {
        Pr.col(i) = P.col(i) - C; // 3*1的向量之差，减去质心
    }
}

// 本部分代码的核心：根据输入的三组匹配对，求解Sim3中的R,t,s
void Sim3Solver::ComputeSim3(cv::Mat &P1, cv::Mat &P2)
{
    // ！！！！！！！这段代码一定要看这篇论文！！！！！！！！！！！
    // Custom implementation of:
    // Horn 1987, Closed-form solution of absolute orientataion using unit quaternions

    // Step 1: Centroid and relative coordinates（模型坐标系）
    // 这里只是初始化定义，没有赋值，Pr1和Pr2都是3*3的Mat类型
    cv::Mat Pr1(P1.size(), P1.type()); // Relative coordinates to centroid (set 1)
    cv::Mat Pr2(P2.size(), P2.type()); // Relative coordinates to centroid (set 2)
    cv::Mat O1(3, 1, Pr1.type());      // Centroid of P1
    cv::Mat O2(3, 1, Pr2.type());      // Centroid of P2

    // O1和O2分别为P1和P2矩阵中3D点的质心
    // Pr1和Pr2为减去质心后的3D点
    ComputeCentroid(P1, Pr1, O1);
    ComputeCentroid(P2, Pr2, O2);

    // Step 2: Compute M matrix
    // 原本是3*1与1*3矩阵得到3*3矩阵，然后相加
    // 这里直接简化计算流程，3*3与3*3矩阵相乘结果仍然是3*3矩阵
    cv::Mat M = Pr2 * Pr1.t();

    // Step 3: Compute N matrix
    double N11, N12, N13, N14, N22, N23, N24, N33, N34, N44;

    cv::Mat N(4, 4, P1.type());

    // 以下按照PPT里的构造得到，本质是因为求解对象为四元数，所以不用M做SVD分解
    N11 = M.at<float>(0, 0) + M.at<float>(1, 1) + M.at<float>(2, 2);
    N12 = M.at<float>(1, 2) - M.at<float>(2, 1);
    N13 = M.at<float>(2, 0) - M.at<float>(0, 2);
    N14 = M.at<float>(0, 1) - M.at<float>(1, 0);
    N22 = M.at<float>(0, 0) - M.at<float>(1, 1) - M.at<float>(2, 2);
    N23 = M.at<float>(0, 1) + M.at<float>(1, 0);
    N24 = M.at<float>(2, 0) + M.at<float>(0, 2);
    N33 = -M.at<float>(0, 0) + M.at<float>(1, 1) - M.at<float>(2, 2);
    N34 = M.at<float>(1, 2) + M.at<float>(2, 1);
    N44 = -M.at<float>(0, 0) - M.at<float>(1, 1) + M.at<float>(2, 2);

    N = (cv::Mat_<float>(4, 4) << N11, N12, N13, N14,
         N12, N22, N23, N24,
         N13, N23, N33, N34,
         N14, N24, N34, N44);

    // Step 4: Eigenvector of the highest eigenvalue
    cv::Mat eval, evec;

    // 四元数的解为SVD分解最大奇异值对应的特征向量
    cv::eigen(N, eval, evec); //evec[0] is the quaternion of the desired rotation

    // N矩阵最大特征值（第一个特征值）对应特征向量就是要求的四元数（q0 q1 q2 q3）
    // 注意：由于此处四元数没有进行归一化，因此转化公式与常见的有所不同，需要先算norm
    // 将(q1 q2 q3)放入vec行向量，vec就是四元数旋转轴乘以sin(ang/2)
    cv::Mat vec(1, 3, evec.type());
    (evec.row(0).colRange(1, 4)).copyTo(vec); //extract imaginary part of the quaternion (sin*axis)

    // Rotation angle. sin is the norm of the imaginary part, cos is the real part
    // ang为旋转的角度theta/2，所以真实的旋转角度为2*ang
    double ang = atan2(norm(vec), evec.at<float>(0, 0));

    // 角轴表示即为theta*n，由于需要归一化，所以要除以norm
    vec = 2 * ang * vec / norm(vec); // Angle-axis representation. quaternion angle is the half

    mR12i.create(3, 3, P1.type());

    // 转化成旋转矩阵的表示
    cv::Rodrigues(vec, mR12i); // computes the rotation matrix from angle-axis

    // Step 5: Rotate set 2
    cv::Mat P3 = mR12i * Pr2;

    // Step 6: Scale
    if (!mbFixScale)
    {
        // 论文中还有一个求尺度的公式，p632右中的位置，那个公式不用考虑旋转
        // 因此下面的尺度求解方法并不是最好的
        double nom = Pr1.dot(P3); // D = ∑ pi'^T * R * qi'
        cv::Mat aux_P3(P3.size(), P3.type());
        aux_P3 = P3;
        cv::pow(P3, 2, aux_P3); // 计算结果保存在aux_P3中
        double den = 0;

        // Sq = ∑ ||R * qi'||²
        for (int i = 0; i < aux_P3.rows; i++)
        {
            for (int j = 0; j < aux_P3.cols; j++)
            {
                den += aux_P3.at<float>(i, j);
            }
        }

        ms12i = nom / den;
    }
    else
        ms12i = 1.0f;

    // Step 7: Translation
    // t为质心之间经过尺度与旋转变换后的距离
    mt12i.create(1, 3, P1.type());
    mt12i = O1 - ms12i * mR12i * O2;

    // Step 8: Transformation
    // Step 8.1 T12
    mT12i = cv::Mat::eye(4, 4, P1.type()); // 注意是eye矩阵，保证了左下为0，右下为1

    cv::Mat sR = ms12i * mR12i;

    //         |sR t|
    // mT12i = | 0 1|
    sR.copyTo(mT12i.rowRange(0, 3).colRange(0, 3));
    mt12i.copyTo(mT12i.rowRange(0, 3).col(3));

    // Step 8.2 T21
    mT21i = cv::Mat::eye(4, 4, P1.type());

    cv::Mat sRinv = (1.0 / ms12i) * mR12i.t(); // 尺度倒数，旋转求转置

    sRinv.copyTo(mT21i.rowRange(0, 3).colRange(0, 3));
    cv::Mat tinv = -sRinv * mt12i; // 偏移结果推导可得
    tinv.copyTo(mT21i.rowRange(0, 3).col(3));
}

// RANSAC中的第三步，需要根据阈值确定Inlier的数量
void Sim3Solver::CheckInliers()
{
    vector<cv::Mat> vP1im2, vP2im1;
    Project(mvX3Dc2, vP2im1, mT12i, mK1); // 把2系中的3D经过Sim3变换(mT12i)到1系中计算重投影坐标
    Project(mvX3Dc1, vP1im2, mT21i, mK2); // 把1系中的3D经过Sim3变换(mT21i)到2系中计算重投影坐标

    mnInliersi = 0;

    for (size_t i = 0; i < mvP1im1.size(); i++)
    {
        cv::Mat dist1 = mvP1im1[i] - vP2im1[i]; // 求重投影误差
        cv::Mat dist2 = vP1im2[i] - mvP2im2[i];

        const float err1 = dist1.dot(dist1); // 平方
        const float err2 = dist2.dot(dist2);

        // 给定了阈值
        if (err1 < mvnMaxError1[i] && err2 < mvnMaxError2[i])
        {
            mvbInliersi[i] = true;
            mnInliersi++;
        }
        else
            mvbInliersi[i] = false;
    }
}

// 下面三个函数是用于外界从类里获得求解结果的
cv::Mat Sim3Solver::GetEstimatedRotation()
{
    return mBestRotation.clone();
}

cv::Mat Sim3Solver::GetEstimatedTranslation()
{
    return mBestTranslation.clone();
}

float Sim3Solver::GetEstimatedScale()
{
    return mBestScale;
}

// CheckInliers里用到，用于计算重投影坐标，与下面的FromCameraToImage类似，多了R,t的转换
// 没有尺度s的原因是投影结果为2d点，与尺度无关
void Sim3Solver::Project(const vector<cv::Mat> &vP3Dw, vector<cv::Mat> &vP2D, cv::Mat Tcw, cv::Mat K)
{
    cv::Mat Rcw = Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = Tcw.rowRange(0, 3).col(3);
    const float &fx = K.at<float>(0, 0);
    const float &fy = K.at<float>(1, 1);
    const float &cx = K.at<float>(0, 2);
    const float &cy = K.at<float>(1, 2);

    vP2D.clear();
    vP2D.reserve(vP3Dw.size());

    for (size_t i = 0, iend = vP3Dw.size(); i < iend; i++)
    {
        cv::Mat P3Dc = Rcw * vP3Dw[i] + tcw;
        const float invz = 1 / (P3Dc.at<float>(2));
        const float x = P3Dc.at<float>(0) * invz;
        const float y = P3Dc.at<float>(1) * invz;

        vP2D.push_back((cv::Mat_<float>(2, 1) << fx * x + cx, fy * y + cy));
    }
}

// 最基本的相机坐标系到图像坐标系的转换
// Z*[x y 1]' = K*[X Y Z]'
void Sim3Solver::FromCameraToImage(const vector<cv::Mat> &vP3Dc, vector<cv::Mat> &vP2D, cv::Mat K)
{
    const float &fx = K.at<float>(0, 0);
    const float &fy = K.at<float>(1, 1);
    const float &cx = K.at<float>(0, 2);
    const float &cy = K.at<float>(1, 2);

    vP2D.clear();
    vP2D.reserve(vP3Dc.size());

    for (size_t i = 0, iend = vP3Dc.size(); i < iend; i++)
    {
        const float invz = 1 / (vP3Dc[i].at<float>(2));
        const float x = vP3Dc[i].at<float>(0) * invz;
        const float y = vP3Dc[i].at<float>(1) * invz;

        vP2D.push_back((cv::Mat_<float>(2, 1) << fx * x + cx, fy * y + cy));
    }
}

} // namespace ORB_SLAM2
