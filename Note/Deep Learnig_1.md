# Deep Learnig_1

## _Coursera, Andrew Ng



- ReLu

  : Rectified linear units

  : 0을 최대값으로 한다.

- Size(x) -> ( ) -> Price(y)  == 1개의 신경세포 == (작은 신경망)

- 여러 신경세포들로 이루어진 모습(Like 레고블럭) == (큰 신경망)

- X 값들과 Y 값 사이는 '트레이닝 센터'

- 트레이닝 센터: ReLu or 비선형 함수가 있음

- X 값과 Y 값을 트레이닝 센터에 넣으면, 가운데 부분은 알아서 해결된다.

- Input layers의 X 값들을 바탕으로 Hidden units을 거쳐 Y 값을 예측한다.

- X, Y 트레이닝 잘된 X, Y가 있으면, X에서 Y를 그리는 함수를 굉장히 잘 파악한다.

<br>

## 지도 학습

### Supervised Learning with Neural Networks

| Input(X)       | Output(Y)          | Application               |
| -------------- | ------------------ | ------------------------- |
| Home features  | Price              | Real Estate (Standard NN) |
| Ad, user info. | Click on ad? (0/1) | Online Ad. (Standard NN)  |
| Image          | Object(1,...,1000) | Photo tagging (CNN)       |
| Audio          | Text transcript    | Speech recognition (RNN)  |
| English        | Korean             | Machine translation (RNN) |

- 신경망이 X, Y 값을 똑똑하게 선정하는 것만으로도 상당한 가치를 지님
- Standard NN: 기본적인 X, Hidden layers, Y 값을 가진 구조
- Convolutional NN: Image data
- Recurrent NN: 시간적 요소를 다루는 1차원적인 데이터

<br>

## Superised Learning

- Structured Data
  - 데이터의 데이터 베이스들
  - 경제적으로 이익이 큼
- Unstructured Data
  - Audio, Image, Text 
  - 강의에선 이 데이터로 알고리즘을 설명할 것
  - 비정형적인 데이터를 설명할 수 있다는 것에 큰 매력이 있음

<br>

## Deep Learning이 뜬 원인!!

- X축: Amount of labeled data
- Y축: Performance
- labeled data: training 예시 (X 입력값, y 레이블 트레이닝 예시)
- m(가로축): training size, training 예시

<br>

- Data
- Computation
- Algorithms

<br>

- 시그모이드 함수 => ReLu 함수
  - ReLu, 기울기 모든 양수에 '1'
  - ReLu, Gradiant descent 기울기 강화!

<br>

- Idea, Code, Experiment 가 Circle로 순환
  - New Network training ↑록, Circle ↑, 생산성 큰 차이!
  - 10 min이 걸릴 수도, 1 day이 걸릴 수도, 1 month이 걸릴 수도 있다. 