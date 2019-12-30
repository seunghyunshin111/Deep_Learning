# Deep Learning_2

## _Coursera, Andrew Ng

## Basics of Neural Network Programming

## Binary Classification

- 신경망 프로그래밍 기초

- 기존: m개의 학습 표본을 가진 training set(학습 표본)이 있으면, m개의 학습 표본에 대해 for문을 돌리면서 하나씩 training set(학습 표본)을 처리 해왔을 것

- Neural Network: 신경망을 구현할 때는 전체 training set를 돌릴 때 for loop를 사용하지 않기로 한다.

- 신경망으로 계산할 때 보통 순전파(순방향 경로), 역전파 계산법을 사용한다.

- 왜 순전파와 역전파를 이용해 계산을 하는지 소개할 것

- 로지스틱 회귀를 통해 아이디어를 제공할 것

  <br>

  - Logistic Regression == Binary Classification (이진 분류를 위한 알고리즘)
  - 1 (Cat) vs - (Non Cat)

  - y: Ouput labels, 출력 레이블
  - 3개의 분리된 행렬