## Mismatched train/test distribution

- ### Training set

  Cat pictures from webpages,  훈련이 잘 된 고화질의 Cat 사진

<br>

- ### Dev/test sets:

  Cat pictures from users using your app, 화질이 좋지 않거나, 핸드폰으로 캐쥬얼하게 찍은 사진일 가능성 높음

<br>

- 그러므로 데이터의 분포도가 전혀 다를 수 있다
- Not having a test set might be okay. (Only dev set.)
- 테스트 세트의 목표는 선택한 마지막 네트워크에 대해서 바이어스 없는 성능 추정치를 제공하는 것이기 때문에, 바이어스 없는 추정치가 필요치 않은 경우, 테스트 세트가 없어도 된다.

<br>

## Bias / Variance

- 편향과 편차에 대한 균형에 대한 토론은 줄었다. 
- high bias: underfitting (대각선)
- high variance: outfitting
- just right: classifier, 중간 정도의 복잡성

<br>

- Bigger network, More data => Lower bias, Lower variance