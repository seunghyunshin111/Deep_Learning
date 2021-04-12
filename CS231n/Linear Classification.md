## Linear Classification

In the last section we introduced the problem of Image Classification, which is the task of assigning a single label to an image from a fixed set of categories. Morever, we described the k-Nearest Neighbor (kNN) classifier which labels images by comparing them to (annotated) images from the training set. As we saw, kNN has a number of disadvantages:

- The classifier must *remember* all of the training data and store it for future comparisons with the test data. This is space inefficient because datasets may easily be gigabytes in size.
- Classifying a test image is expensive since it requires a comparison to all training images.

**Overview**. We are now going to develop a more powerful approach to image classification that we will eventually naturally extend to entire Neural Networks and Convolutional Neural Networks. The approach will have two major components: a **score function** that maps the raw data to class scores, and a **loss function** that quantifies the agreement between the predicted scores and the ground truth labels. We will then cast this as an optimization problem in which we will minimize the loss function with respect to the parameters of the score function.





마지막 섹션에서는 고정 된 범주 집합의 이미지에 단일 레이블을 할당하는 작업인 이미지 분류 문제를 소개했습니다. 또한 훈련 세트의 (주석이 있는) 이미지와 비교하여 이미지에 레이블을 지정하는 kNN (K-Nearest Neighbor) 분류기를 설명했습니다. 우리가 보았듯이 kNN에는 여러 가지 단점이 있습니다.

-분류기는 모든 학습 데이터를 * 기억 *하고 향후 테스트 데이터와 비교할 수 있도록 저장해야합니다. 데이터 세트의 크기가 쉽게 기가 바이트가 될 수 있기 때문에 이것은 공간 비효율적입니다.
-테스트 이미지 분류는 모든 학습 이미지와 비교해야 하므로 비용이 많이 듭니다.

** 개요 **. 이제 우리는 이미지 분류에 대한보다 강력한 접근 방식을 개발할 것이며 결국에는 전체 신경망과 컨볼루션 신경망으로 자연스럽게 확장 될 것입니다. 이 접근 방식에는 원시 데이터를 클래스 점수에 매핑하는 ** 점수 함수 **와 예측 점수와 실측 값 레이블 간의 일치를 정량화하는 ** 손실 함수 **의 두 가지 주요 구성 요소가 있습니다. 그런 다음 이를 점수 함수의 매개 변수에 대한 손실 함수를 최소화하는 최적화 문제로 캐스팅합니다.

=> KNN은 분류 모델로, 학습한 후 향후 테스트 데이터로 이미지를 분류하는 접근 방식이다. 해당 분류기는 모든 학습 데이터를 기억하고, 향후 테스트 데이터와 비교하기 때문에 데이터 세트의 크기가 기본적으로 크다. 따라서 테스트 이미지 분류 시, 이미지 분류는 모든 학습 이미지와 대조해보기 때문에 비용이 높게 든다. 



### Parameterized mapping from images to label scores

The first component of this approach is to define the score function that maps the pixel values of an image to confidence scores for each class. We will develop the approach with a concrete example. As before, let’s assume a training dataset of images xi∈RDxi∈RD, each associated with a label yiyi. Here i=1…Ni=1…N and yi∈1…Kyi∈1…K. That is, we have **N** examples (each with a dimensionality **D**) and **K** distinct categories. For example, in CIFAR-10 we have a training set of **N** = 50,000 images, each with **D** = 32 x 32 x 3 = 3072 pixels, and **K** = 10, since there are 10 distinct classes (dog, cat, car, etc). We will now define the score function f:RD↦RKf:RD↦RK that maps the raw image pixels to class scores.

**Linear classifier.** In this module we will start out with arguably the simplest possible function, a linear mapping:

f(xi,W,b)=Wxi+bf(xi,W,b)=Wxi+b

In the above equation, we are assuming that the image xixi has all of its pixels flattened out to a single column vector of shape [D x 1]. The matrix **W** (of size [K x D]), and the vector **b** (of size [K x 1]) are the **parameters** of the function. In CIFAR-10, xixi contains all pixels in the i-th image flattened into a single [3072 x 1] column, **W** is [10 x 3072] and **b** is [10 x 1], so 3072 numbers come into the function (the raw pixel values) and 10 numbers come out (the class scores). The parameters in **W** are often called the **weights**, and **b** is called the **bias vector** because it influences the output scores, but without interacting with the actual data xixi. However, you will often hear people use the terms *weights* and *parameters* interchangeably.

There are a few things to note:

- First, note that the single matrix multiplication WxiWxi is effectively evaluating 10 separate classifiers in parallel (one for each class), where each classifier is a row of **W**.
- Notice also that we think of the input data (xi,yi)(xi,yi) as given and fixed, but we have control over the setting of the parameters **W,b**. Our goal will be to set these in such way that the computed scores match the ground truth labels across the whole training set. We will go into much more detail about how this is done, but intuitively we wish that the correct class has a score that is higher than the scores of incorrect classes.
- An advantage of this approach is that the training data is used to learn the parameters **W,b**, but once the learning is complete we can discard the entire training set and only keep the learned parameters. That is because a new test image can be simply forwarded through the function and classified based on the computed scores.
- Lastly, note that classifying the test image involves a single matrix multiplication and addition, which is significantly faster than comparing a test image to all training images.

> Foreshadowing: Convolutional Neural Networks will map image pixels to scores exactly as shown above, but the mapping ( f ) will be more complex and will contain more parameters.



### 이미지에서 라벨 점수로 매개 변수화 된 매핑

이 접근 방식의 첫 번째 구성 요소는 이미지의 픽셀 값을 각 클래스의 신뢰도 점수에 매핑하는 점수 함수를 정의하는 것입니다. 구체적인 예를 들어 접근 방식을 개발할 것입니다. 이전과 마찬가지로 각각 레이블 yiyi와 연결된 이미지 xi∈RDxi∈RD의 학습 데이터 세트를 가정 해 보겠습니다. 여기서 i = 1… Ni = 1… N과 yi∈1… Kyi∈1… K. 즉, ** N ** 개의 예 (각각 차원이 ** D ** 인 경우)와 ** K ** 개의 개별 카테고리가 있습니다. 예를 들어 CIFAR-10에는 각각 ** D ** = 32 x 32 x 3 = 3072 픽셀이고 ** K ** = 10 인 ** N ** = 50,000 개의 이미지로 구성된 학습 세트가 있습니다. 10 개의 다른 클래스 (개, 고양이, 자동차 등)가 있습니다. 이제 원시 이미지 픽셀을 클래스 점수에 매핑하는 점수 함수 f : RD↦RKf : RD↦RK를 정의합니다.

** 선형 분류기. **이 모듈에서는 가장 간단한 함수인 선형 매핑으로 시작합니다.

f (xi, W, b) = Wxi + bf (xi, W, b) = Wxi + b

위의 방정식에서 우리는 이미지 xixi에 모든 픽셀이 [D x 1] 모양의 단일 열 벡터로 평면화되었다고 가정합니다. 행렬 ** W ** (크기 [K x D]) 및 벡터 ** b ** (크기 [K x 1])는 함수의 ** 매개 변수 **입니다. CIFAR-10에서 xixi는 단일 [3072 x 1] 열로 병합 된 i 번째 이미지의 모든 픽셀을 포함합니다. ** W **는 [10 x 3072]이고 ** b **는 [10 x 1]입니다. 따라서 3072 개의 숫자가 함수 (원시 픽셀 값)에 들어오고 10 개의 숫자 (클래스 점수)가 나옵니다. ** W **의 매개 변수는 종종 ** 가중치 **라고하며 ** b **는 출력 점수에 영향을 주지만 실제 데이터 xixi와 상호 작용하지 않기 때문에 ** 바이어스 벡터 **라고합니다. 그러나 사람들이 * weights *와 * parameters *라는 용어를 같은 의미로 사용하는 경우가 종종 있습니다.


몇 가지 참고할 사항이 있습니다.

-첫째, 단일 행렬 곱셈 WxiWxi는 10 개의 개별 분류기를 병렬로 (각 클래스에 하나씩) 효과적으로 평가하고 있으며, 여기서 각 분류기는 ** W **의 행입니다.
-또한 입력 데이터 (xi, yi) (xi, yi)를 주어진 고정 된 것으로 생각하지만 매개 변수 ** W, b **의 설정을 제어 할 수 있습니다. 우리의 목표는 계산 된 점수가 전체 훈련 세트에 걸쳐 지상 실측 레이블과 일치하는 방식으로 이를 설정하는 것입니다. 우리는 이것이 어떻게 수행되는지에 대해 훨씬 더 자세히 설명 할 것이지만 직관적으로 올바른 클래스의 점수가 잘못된 클래스의 점수보다 높기를 바랍니다.
-이 접근 방식의 장점은 학습 데이터가 매개 변수 ** W, b **를 학습하는 데 사용된다는 것입니다. 하지만 학습이 완료되면 전체 학습 세트를 버리고 학습 된 매개 변수만 유지할 수 있습니다. 새로운 테스트 이미지는 단순히 함수를 통해 전달되고 계산 된 점수에 따라 분류될 수 있기 때문입니다.
-마지막으로, 테스트 이미지를 분류하는 데는 단일 행렬 곱셈과 덧셈이 포함되므로 테스트 이미지를 모든 학습 이미지와 비교하는 것보다 훨씬 빠릅니다.

> Convolutional Neural Networks는 위에 표시된대로 정확하게 이미지 픽셀을 점수에 매핑하지만 매핑 (f)은 더 복잡하고 더 많은 매개 변수를 포함합니다.

=> 매개변수 w(가중치), b의 설정을 통해 최종 분류기로 도달됩니다. 
학습 데이터가 매개변수 w, b를 학습하는데 사용된다는 것이 중요 포인트입니다. 



### Interpreting a linear classifier

Notice that a linear classifier computes the score of a class as a weighted sum of all of its pixel values across all 3 of its color channels. Depending on precisely what values we set for these weights, the function has the capacity to like or dislike (depending on the sign of each weight) certain colors at certain positions in the image. For instance, you can imagine that the “ship” class might be more likely if there is a lot of blue on the sides of an image (which could likely correspond to water). You might expect that the “ship” classifier would then have a lot of positive weights across its blue channel weights (presence of blue increases score of ship), and negative weights in the red/green channels (presence of red/green decreases the score of ship).

![img](https://cs231n.github.io/assets/imagemap.jpg)



### 선형 분류기 해석

선형 분류기는 3 개의 색상 채널 모두에서 모든 픽셀 값의 가중 합계로 클래스의 점수를 계산합니다. 이러한 가중치에 대해 설정 한 값에 따라 함수는 이미지의 특정 위치에서 특정 색상을 좋아하거나 싫어할 수 있습니다 (각 가중치의 기호에 따라 다름). 예를 들어, 이미지의 측면에 파란색이 많이있는 경우 (물에 해당 할 수 있음) "ship"클래스가 더 많을 수 있다고 상상할 수 있습니다. 그러면 "ship"분류 기가 파란색 채널 가중치 (파란색의 존재는 배의 점수를 증가)에 걸쳐 많은 양의 가중치를 가지며, 적색 / 녹색 채널의 음의 가중치 (적색 / 녹색의 존재는 점수를 감소시킵니다)를 기대할 수 있습니다.

=> 각 3가지 색상 채널은 각각 다른 이미지 분류 카테고리를 의미합니다. 각 이미지의 부분 가중 합계를 통해 최종 클래스의 점수를 계싼합니다. 가중치는 이미지의 특정 위치에서 특정 특성에 해당하는 신호에 따라 가중치를 크기를 부여합니다.

An example of mapping an image to class scores. For the sake of visualization, we assume the image only has 4 pixels (4 monochrome pixels, we are not considering color channels in this example for brevity), and that we have 3 classes (red (cat), green (dog), blue (ship) class). (Clarification: in particular, the colors here simply indicate 3 classes and are not related to the RGB channels.) We stretch the image pixels into a column and perform matrix multiplication to get the scores for each class. Note that this particular set of weights W is not good at all: the weights assign our cat image a very low cat score. In particular, this set of weights seems convinced that it's looking at a dog.

이미지를 클래스 점수에 매핑하는 예입니다. 시각화를 위해 이미지에 4 개의 픽셀 (흑백 픽셀 4 개, 간결성을 위해 예제에서는 색상 채널을 고려하지 않음) 만 있고 3 개의 클래스 (빨간색 (고양이), 녹색 (개), 파란색)가 있다고 가정합니다. (선박) 클래스). (설명 : 특히 여기의 색상은 단순히 3 개의 클래스를 나타내며 RGB 채널과 관련이 없습니다.) 이미지 픽셀을 열로 늘리고 행렬 곱셈을 수행하여 각 클래스의 점수를 얻습니다. 이 특정 가중치 세트 W는 전혀 좋지 않습니다. 가중치는 고양이 이미지에 매우 낮은 고양이 점수를 할당합니다. 특히, 이 가중치 세트는 개를 보고 있다고 확신하는 것 같습니다.

=> 최종적으로 가중치 세트 합의 결과를 보고 가장 좋은 수치의 카테고리 분류 값을 통해 해당 이미지의 카테고리를 결정합니다. 



**Analogy of images as high-dimensional points.** Since the images are stretched into high-dimensional column vectors, we can interpret each image as a single point in this space (e.g. each image in CIFAR-10 is a point in 3072-dimensional space of 32x32x3 pixels). Analogously, the entire dataset is a (labeled) set of points.

Since we defined the score of each class as a weighted sum of all image pixels, each class score is a linear function over this space. We cannot visualize 3072-dimensional spaces, but if we imagine squashing all those dimensions into only two dimensions, then we can try to visualize what the classifier might be doing:

![img](https://cs231n.github.io/assets/pixelspace.jpeg)

Cartoon representation of the image space, where each image is a single point, and three classifiers are visualized. Using the example of the car classifier (in red), the red line shows all points in the space that get a score of zero for the car class. The red arrow shows the direction of increase, so all points to the right of the red line have positive (and linearly increasing) scores, and all points to the left have a negative (and linearly decreasing) scores.

As we saw above, every row of WW is a classifier for one of the classes. The geometric interpretation of these numbers is that as we change one of the rows of WW, the corresponding line in the pixel space will rotate in different directions. The biases bb, on the other hand, allow our classifiers to translate the lines. In particular, note that without the bias terms, plugging in xi=0xi=0 would always give score of zero regardless of the weights, so all lines would be forced to cross the origin.



** 이미지를 고차원 점으로 분석합니다. ** 이미지가 고차원 열 벡터로 확장되므로 각 이미지를 이 공간의 단일 점으로 해석 할 수 있습니다 (예 : CIFAR-10의 각 이미지는 3072의 점입니다. 32x32x3 픽셀의 차원 공간). 마찬가지로 전체 데이터 세트는 (레이블이 지정된) 포인트 세트입니다.

각 클래스의 점수를 모든 이미지 픽셀의 가중치 합계로 정의 했으므로 각 클래스 점수는 이 공간에 대한 선형 함수입니다. 3072 차원 공간을 시각화 할 수는 없지만 모든 차원을 2 차원으로 압축하는 것을 상상한다면 분류기가 수행 할 작업을 시각화 할 수 있습니다.
각 이미지가 단일 포인트이고 세 개의 분류기가 시각화되는 이미지 공간의 만화 표현. 자동차 분류기 (빨간색)의 예를 사용하면 빨간색 선은 자동차 등급에 대해 점수가 0 인 공간의 모든 점을 표시합니다. 빨간색 화살표는 증가 방향을 나타내므로 빨간색 선의 오른쪽에있는 모든 포인트는 양수 (및 선형 적으로 증가) 점수를 가지며 왼쪽에 있는 모든 포인트는 음수 (및 선형 적으로 감소) 점수를 갖습니다.

위에서 보았 듯이 W의 모든 행은 클래스 중 하나에 대한 분류입니다. 이 숫자의 기하학적 해석은 W의 행 중 하나를 변경하면 픽셀 공간의 해당 선이 다른 방향으로 회전한다는 것입니다. 반면 편향 b는 분류기가 라인을 번역 할 수 있도록합니다. 특히 편향 항이 없으면 xi = 0을 연결하면 가중치에 관계 없이 항상 점수가 0이 되므로 모든 선이 원점을 가로지르도록 강제됩니다.

=> 이미지를 각 가중치 합계에 따라 분류하게 됩니다. 



**Interpretation of linear classifiers as template matching.** Another interpretation for the weights WW is that each row of WW corresponds to a *template* (or sometimes also called a *prototype*) for one of the classes. The score of each class for an image is then obtained by comparing each template with the image using an *inner product* (or *dot product*) one by one to find the one that “fits” best. With this terminology, the linear classifier is doing template matching, where the templates are learned. Another way to think of it is that we are still effectively doing Nearest Neighbor, but instead of having thousands of training images we are only using a single image per class (although we will learn it, and it does not necessarily have to be one of the images in the training set), and we use the (negative) inner product as the distance instead of the L1 or L2 distance.

![img](https://cs231n.github.io/assets/templates.jpg)

Skipping ahead a bit: Example learned weights at the end of learning for CIFAR-10. Note that, for example, the ship template contains a lot of blue pixels as expected. This template will therefore give a high score once it is matched against images of ships on the ocean with an inner product.

Additionally, note that the horse template seems to contain a two-headed horse, which is due to both left and right facing horses in the dataset. The linear classifier *merges* these two modes of horses in the data into a single template. Similarly, the car classifier seems to have merged several modes into a single template which has to identify cars from all sides, and of all colors. In particular, this template ended up being red, which hints that there are more red cars in the CIFAR-10 dataset than of any other color. The linear classifier is too weak to properly account for different-colored cars, but as we will see later neural networks will allow us to perform this task. Looking ahead a bit, a neural network will be able to develop intermediate neurons in its hidden layers that could detect specific car types (e.g. green car facing left, blue car facing front, etc.), and neurons on the next layer could combine these into a more accurate car score through a weighted sum of the individual car detectors.

** 선형 분류기를 템플릿 일치로 해석 ** 가중치 W에 대한 또 다른 해석은 W의 각 행이 클래스 중 하나에 대한 * 템플릿 * (또는 * 프로토 타입 *이라고도 함)에 해당한다는 것입니다. 이미지에 대한 각 클래스의 점수는 내적을 사용하여 각 템플릿을 이미지와 비교하여 가장 "적합한" 항목을 찾아서 얻습니다. 이 용어를 사용하여 선형 분류기는 템플릿을 학습하는 템플릿 일치를 수행합니다. 이를 생각하는 또 다른 방법은 우리가 여전히 Nearest Neighbor를 효과적으로 수행하고 있지만 수천 개의 훈련 이미지를 갖는 대신 클래스 당 하나의 이미지만 사용한다는 것입니다 (우리가 학습 할 것이지만 반드시 다음 중 하나일 필요는 없습니다.) 학습 세트의 이미지), L1 또는 L2 거리 대신 (음수) 내적을 거리로 사용합니다.

조금 앞으로 건너 뛰기 : CIFAR-10 학습이 끝날 때 학습 된 가중치 예제. 예를 들어 ship 템플릿에는 예상대로 많은 파란색 픽셀이 포함되어 있습니다. 따라서 이 템플릿은 내부 제품이있는 바다의 선박 이미지와 일치하면 높은 점수를 제공합니다.

또한 말 템플릿에 머리가 두 개인 말이 포함되어 있는 것 같습니다. 이는 데이터 세트에서 왼쪽과 오른쪽을 향하는 말 모두 때문입니다. 선형 분류기는 데이터에 있는 이 두 가지 말 모드를 단일 템플릿으로 * 병합 *합니다. 마찬가지로, 자동차 분류기는 여러 모드를 모든 면에서 모든 색상의 자동차를 식별해야하는 단일 템플릿으로 병합한 것으로 보입니다. 특히, 이 템플릿은 결국 빨간색이 되어 CIFAR-10 데이터 세트에 다른 색상보다 더 많은 빨간색 자동차가 있음을 나타냅니다. 선형 분류기는 너무 약해서 다른 색상의 자동차를 적절하게 설명 할 수 없지만 나중에 보게 될 신경망을 통해 작업을 수행 할 수 있습니다. 조금 앞을 내다 보면 신경망은 특정 자동차 유형 (예 : 왼쪽을 향한 녹색 자동차, 앞을 향한 파란색 자동차 등)을 감지 할 수 있는 숨겨진 계층에서 중간 뉴런을 개발할 수 있으며 다음 계층의 뉴런은 이들을 결합 할 수 있습니다. 개별 자동차 감지기의 가중치 합계를 통해 더 정확한 자동차 점수로 변환합니다.


=> 선형 분류기를 사용하여 템플릿 일치를 수행합니다. 테스트 이미지에 대한 각 클래스 점수 합은 내적을 사용하여 각 템플릿을 이미지와 비교하여 최종적으로 가장 적합한 항목을 설정합니다. 이 방법은 여전히 Nearest Neighbor를 효과적으로 수행한다고 볼 수 있지만, 한 가지 특징은 / 수천 개의 훈련 이미지를 갖는 위의 단점 대신 / 클래스 당 하나의 이미지만 사용한다는 점에서 이전 단점을 극복한 방법입니다. 

=> 선형 분류기는 너무 약하기 때문에 빨간색 자동차의 템플릿이라면 / 다른 색상의 자동차를 적절하게 설명 할 수는 없다. 다만, 향후 신경망을 통해서는 해당 작업을 보완할 수 있다. 신경망은 특정 자동차 유형 (색상 / 각도 등)을 감지할 수 있다. 이는 중간 뉴런에서 이들을 판단하는 것을 개발할 수 있기 때문이다. 따라서 개별 자동차 감지기의 가중치 합계를 통해 결국 더 정확한 자동차 점수를 변환합니다. 



**Bias trick.** Before moving on we want to mention a common simplifying trick to representing the two parameters W,bW,b as one. Recall that we defined the score function as:

f(xi,W,b)=Wxi+bf(xi,W,b)=Wxi+b

As we proceed through the material it is a little cumbersome to keep track of two sets of parameters (the biases bb and weights WW) separately. A commonly used trick is to combine the two sets of parameters into a single matrix that holds both of them by extending the vector xixi with one additional dimension that always holds the constant 11 - a default *bias dimension*. With the extra dimension, the new score function will simplify to a single matrix multiply:

f(xi,W)=Wxif(xi,W)=Wxi

With our CIFAR-10 example, xixi is now [3073 x 1] instead of [3072 x 1] - (with the extra dimension holding the constant 1), and WW is now [10 x 3073] instead of [10 x 3072]. The extra column that WW now corresponds to the bias bb. An illustration might help clarify:

![img](https://cs231n.github.io/assets/wb.jpeg)

Illustration of the bias trick. Doing a matrix multiplication and then adding a bias vector (left) is equivalent to adding a bias dimension with a constant of 1 to all input vectors and extending the weight matrix by 1 column - a bias column (right). Thus, if we preprocess our data by appending ones to all vectors we only have to learn a single matrix of weights instead of two matrices that hold the weights and the biases.

** 바이어스 트릭. ** 계속 진행하기 전에 두 개의 매개 변수 W, bW, b를 하나로 표현하는 일반적인 단순화 트릭을 언급하고자합니다. 점수 함수를 다음과 같이 정의했습니다.

f (xi, W, b) = Wxi + bf (xi, W, b) = Wxi + b

자료를 진행하면서 두 세트의 매개 변수 (바이어스 b 및 가중치 W)를 별도로 추적하는 것은 약간 번거롭습니다. 일반적으로 사용되는 트릭은 벡터 xi를 항상 상수 1 (기본 * 바이어스 차원 *)을 유지하는 하나의 추가 차원으로 확장하여 두 세트의 매개 변수를 둘 다 유지하는 단일 행렬로 결합하는 것입니다. 추가 차원을 사용하면 새로운 점수 함수는 단일 행렬 곱셈으로 단순화됩니다.

f (xi, W) = Wxif (xi, W) = Wxi

CIFAR-10 예제에서 xi는 이제 [3072 x 1] 대신 [3073 x 1]입니다 (추가 차원은 상수 1을 유지함). W는 이제 [10 x 3072] 대신 [10 x 3073]입니다. W가 이제 바이어스 b에 해당하는 추가 열입니다. 다음과 같은 설명이 도움이 될 수 있습니다.

바이어스 트릭의 그림. 행렬 곱셈을 수행한 다음 편향 벡터 (왼쪽)를 추가하는 것은 모든 입력 벡터에 상수가 1 인 편향 차원을 추가하고 가중치 행렬을 편향 열 (오른쪽) 1 개씩 확장하는 것과 같습니다. 따라서 모든 벡터에 1을 추가하여 데이터를 전처리하면 가중치와 편향을 포함하는 두 개의 행렬 대신 단일 가중치 행렬만 학습하면 됩니다.


=> 매개 변수를 둘 다 유지하는 단일 행렬로 결합하고 / 추가 차원을 사용하면 새로운 점수 함수는 단일 행렬 곱셈으로 단순화됩니다. 

=> 행렬 곱셈을 수행한 다음 편향 벡터를 추가하는 것은 모든 입력 벡터에 상수가 1인 편향 차원을 추가하고 가중치 행렬을 편향 열 1개씩 확정하는 것과 같다. 따라서 모든 벡터에 1을 추가하여 데이터 전처리를 하면 가중치 / 편향을 포함하는 두 개의 행렬 대신 "단일 가중치 행렬"만 학습하면 된다. (어렵)

**Image data preprocessing.** As a quick note, in the examples above we used the raw pixel values (which range from [0…255]). In Machine Learning, it is a very common practice to always perform normalization of your input features (in the case of images, every pixel is thought of as a feature). In particular, it is important to **center your data** by subtracting the mean from every feature. In the case of images, this corresponds to computing a *mean image* across the training images and subtracting it from every image to get images where the pixels range from approximately [-127 … 127]. Further common preprocessing is to scale each input feature so that its values range from [-1, 1]. Of these, zero mean centering is arguably more important but we will have to wait for its justification until we understand the dynamics of gradient descent.

** 이미지 데이터 전처리. ** 위의 예에서는 원시 픽셀 값 ([0… 255] 범위)을 사용했습니다. 머신러닝에서는 항상 입력 특성의 정규화를 수행하는 것이 매우 일반적인 관행입니다 (이미지의 경우 모든 픽셀이 특성으로 간주 됨). 특히 모든 특성에서 평균을 빼서 ** 데이터 중심 **을 지정하는 것이 중요합니다. 이미지의 경우, 이것은 훈련 이미지에서 * 평균 이미지 *를 계산하고 모든 이미지에서 이를 빼서 픽셀 범위가 약 [-127… 127] 인 이미지를 얻는 것에 해당합니다. 더 일반적인 전처리는 각 입력 특성의 값을 [-1, 1] 범위로 조정하는 것입니다. 이 중 제로 평균 중심화가 더 중요하지만 경사 하강 법의 역할을 이해할 때까지 정당화를 기다려야합니다.

=> 머신러닝에서는 항상 입력 특성의 정규화를 수행하는 것이 일반적인 관행.
=> 모든 특성에서 평균을 빼 데이터 중심을 지정하는 것이 중요하다. 이를 이미지로 대조해보면, 훈련 이미지에서 평균 이미지를 계산하고, 모든 이미지에서 이를 빼 픽셀 범위의 데이터 중심을 지정하는 것이다. 더 일반적으로는 각 입력 특성의 값을 [-1, 1] 범위로 조정하는 것이다.

### Loss function

In the previous section we defined a function from the pixel values to class scores, which was parameterized by a set of weights WW. Moreover, we saw that we don’t have control over the data (xi,yi)(xi,yi) (it is fixed and given), but we do have control over these weights and we want to set them so that the predicted class scores are consistent with the ground truth labels in the training data.

For example, going back to the example image of a cat and its scores for the classes “cat”, “dog” and “ship”, we saw that the particular set of weights in that example was not very good at all: We fed in the pixels that depict a cat but the cat score came out very low (-96.8) compared to the other classes (dog score 437.9 and ship score 61.95). We are going to measure our unhappiness with outcomes such as this one with a **loss function** (or sometimes also referred to as the **cost function** or the **objective**). Intuitively, the loss will be high if we’re doing a poor job of classifying the training data, and it will be low if we’re doing well.

### 손실 함수

이전 섹션에서 우리는 픽셀 값에서 클래스 점수로 함수를 정의했으며, 이는 가중치 W 세트로 매개 변수화되었습니다. 더욱이 우리는 데이터 (xi, yi) (xi, yi) (고정되고 주어짐)에 대한 통제권이 없다는 것을 알았습니다. 그러나 우리는 이러한 가중치를 통제 할 수 있으며 예측 된 클래스 점수는 훈련 데이터의 실측 레이블과 일치합니다.

예를 들어, 고양이의 예제 이미지와 "cat", "dog"및 "ship"클래스에 대한 점수로 돌아가서 해당 예제의 특정 가중치 집합이 전혀 좋지 않음을 확인했습니다. 고양이를 묘사하는 픽셀에서 고양이 점수는 다른 등급 (개 점수 437.9 및 선박 점수 61.95)에 비해 매우 낮았습니다 (-96.8). ** 손실 함수 ** (또는 ** 비용 함수 ** 또는 ** 목표 **라고도 함)와 같은 결과에 대한 불행을 측정 할 것입니다. 직관적으로 학습 데이터 분류 작업을 제대로 수행하지 않으면 손실이 크고 잘 수행하면 손실이 적습니다.


=> 이전 섹션에서 픽셀 값을 통해 클래스 점수로 함수를 정의했다면, 이는 가중치 W 세트로 매개 변수화 되었다. 우리는 가중치를 통제하여 예측된 클래스 점수가 훈련 데이터의 실측 레이블과 일치하게 만들었다.

=> 학습 데이터 분류 작업을 제대로 수행하면 손실이 적다.

#### Multiclass Support Vector Machine loss

There are several ways to define the details of the loss function. As a first example we will first develop a commonly used loss called the **Multiclass Support Vector Machine** (SVM) loss. The SVM loss is set up so that the SVM “wants” the correct class for each image to a have a score higher than the incorrect classes by some fixed margin Δ. Notice that it’s sometimes helpful to anthropomorphise the loss functions as we did above: The SVM “wants” a certain outcome in the sense that the outcome would yield a lower loss (which is good).

Let’s now get more precise. Recall that for the i-th example we are given the pixels of image xixi and the label yiyi that specifies the index of the correct class. The score function takes the pixels and computes the vector f(xi,W)f(xi,W) of class scores, which we will abbreviate to ss (short for scores). For example, the score for the j-th class is the j-th element: sj=f(xi,W)jsj=f(xi,W)j. The Multiclass SVM loss for the i-th example is then formalized as follows:

Li=∑j≠yimax(0,sj−syi+Δ)Li=∑j≠yimax(0,sj−syi+Δ)

#### 다중 클래스 지원 벡터 머신 손실

손실 함수의 세부 사항을 정의하는 방법에는 여러 가지가 있습니다. 첫 번째 예로, ** Multiclass Support Vector Machine ** (SVM) 손실이라고하는 일반적으로 사용되는 손실을 먼저 개발합니다. SVM 손실은 SVM이 각 이미지에 대한 올바른 클래스가 고정 마진 Δ만큼 잘못된 클래스보다 높은 점수를 갖도록 "원"하도록 설정됩니다. 위에서 한 것처럼 손실 함수를 의인화하는 것이 때때로 도움이 됩니다. SVM은 결과가 더 낮은 손실 (좋은)을 산출한다는 의미에서 특정 결과를 "원"합니다.

이제 더 정확하게 봅시다. i 번째 예제의 경우 이미지 xi의 픽셀과 올바른 클래스의 인덱스를 지정하는 레이블 yi가 주어집니다. score 함수는 픽셀을 가져 와서 클래스 점수의 벡터 f (xi, W)를 계산하며,이를 s (점수의 약자)로 축약합니다. 예를 들어, j 번째 클래스의 점수는 j 번째 요소 인 sj = f (xi, W) jsj = f (xi, W) j입니다. i 번째 예제에 대한 다중 클래스 SVM 손실은 다음과 같이 공식화됩니다.

Li = ∑j ≠ yimax (0, sj−syi + Δ) Li = ∑j ≠ yimax (0, sj−syi + Δ)

=> 손실함수 중 하나로 SVM 모델이 있다. SVM 손실은 각 이미지에 대한 올바른 클래스가 고정 마진만큼 잘못된 클래스보다 높은 점수를 갖도록 설정하는 것이다. 

**Example.** Lets unpack this with an example to see how it works. Suppose that we have three classes that receive the scores s=[13,−7,11]s=[13,−7,11], and that the first class is the true class (i.e. yi=0yi=0). Also assume that ΔΔ (a hyperparameter we will go into more detail about soon) is 10. The expression above sums over all incorrect classes (j≠yij≠yi), so we get two terms:

Li=max(0,−7−13+10)+max(0,11−13+10)Li=max(0,−7−13+10)+max(0,11−13+10)

You can see that the first term gives zero since [-7 - 13 + 10] gives a negative number, which is then thresholded to zero with the max(0,−)max(0,−) function. We get zero loss for this pair because the correct class score (13) was greater than the incorrect class score (-7) by at least the margin 10. In fact the difference was 20, which is much greater than 10 but the SVM only cares that the difference is at least 10; Any additional difference above the margin is clamped at zero with the max operation. The second term computes [11 - 13 + 10] which gives 8. That is, even though the correct class had a higher score than the incorrect class (13 > 11), it was not greater by the desired margin of 10. The difference was only 2, which is why the loss comes out to 8 (i.e. how much higher the difference would have to be to meet the margin). In summary, the SVM loss function wants the score of the correct class yiyi to be larger than the incorrect class scores by at least by ΔΔ (delta). If this is not the case, we will accumulate loss.

Note that in this particular module we are working with linear score functions ( f(xi;W)=Wxif(xi;W)=Wxi ), so we can also rewrite the loss function in this equivalent form:

Li=∑j≠yimax(0,wTjxi−wTyixi+Δ)Li=∑j≠yimax(0,wjTxi−wyiTxi+Δ)

where wjwj is the j-th row of WW reshaped as a column. However, this will not necessarily be the case once we start to consider more complex forms of the score function ff.

A last piece of terminology we’ll mention before we finish with this section is that the threshold at zero max(0,−)max(0,−) function is often called the **hinge loss**. You’ll sometimes hear about people instead using the squared hinge loss SVM (or L2-SVM), which uses the form max(0,−)2max(0,−)2 that penalizes violated margins more strongly (quadratically instead of linearly). The unsquared version is more standard, but in some datasets the squared hinge loss can work better. This can be determined during cross-validation.

> The loss function quantifies our unhappiness with predictions on the training set

![img](https://cs231n.github.io/assets/margin.jpg)

The Multiclass Support Vector Machine "wants" the score of the correct class to be higher than all other scores by at least a margin of delta. If any class has a score inside the red region (or higher), then there will be accumulated loss. Otherwise the loss will be zero. Our objective will be to find the weights that will simultaneously satisfy this constraint for all examples in the training data and give a total loss that is as low as possible.

* 예 ** 작동 방식을 확인하기 위해 예제와 함께 압축을 풀어 보겠습니다. s = [13, −7,11] s = [13, −7,11] 점수를받는 세 개의 클래스가 있고 첫 번째 클래스가 실제 클래스 (i.e. yi = 0yi = 0)라고 가정합니다. 또한 Δ (곧 자세히 설명 할 하이퍼 파라미터)가 10이라고 가정합니다. 위의 표현식은 모든 잘못된 클래스 (j ≠ yij ≠ yi)에 대해 합산되므로 두 가지 항을 얻습니다.

Li = max (0, −7−13 + 10) + max (0,11−13 + 10) Li = max (0, −7−13 + 10) + max (0,11−13 + 10)

[-7-13 + 10]이 음수를 제공하므로 첫 번째 항이 0을 제공한다는 것을 알 수 있습니다. 그러면 max (0, −) max (0, −) 함수를 사용하여 0으로 임계 값이 설정됩니다. 정확한 클래스 점수 (13)가 잘못된 클래스 점수 (-7)보다 적어도 마진 10만큼 더 높았기 때문에 이 쌍에 대해 손실이 없습니다. 실제로 차이는 20이었는데, 이는 10보다 훨씬 크지만 SVM 만 차이가 10 이상인지 확인합니다. 여백을 초과하는 추가 차이는 최대 작동으로 0으로 고정됩니다. 두 번째 학기는 8을 제공하는 [11-13 + 10]을 계산합니다. 즉, 올바른 클래스가 잘못된 클래스 (13> 11)보다 높은 점수를 받았지만 원하는 마진인 10만큼 크지 않습니다. 2에 불과했기 때문에 손실이 8로 나옵니다 (즉, 마진을 충족하려면 차이가 얼마나 높아야하는지). 요약하면, SVM 손실 함수는 올바른 클래스 yi의 점수가 잘못된 클래스 점수보다 적어도 Δ (델타)만큼 더 커지기를 원합니다. 그렇지 않으면 손실이 누적됩니다.


==> 결론적으로 SVM 손실 함수는 올바른 클래스의 점수가 잘못된 클래스 점수보다 적어도 델타만큼 더 커져야 합니다. 그렇지 않으면 손실이 누적되는 케이스입니다.



이 특정 모듈에서 선형 점수 함수 (f (xi; W) = Wxif (xi; W) = Wxi)로 작업하고 있으므로 손실 함수를 다음과 같은 형식으로 다시 작성할 수도 있습니다.

Li = ∑j ≠ yimax (0, wTjxi−wTyixi + Δ) Li = ∑j ≠ yimax (0, wjTxi−wyiTxi + Δ)

여기서 wj는 W의 j 번째 행이 열로 재구성 된 것입니다. 그러나 점수 함수 f의 더 복잡한 형태를 고려하기 시작하면 반드시 그런 것은 아닙니다.

이 섹션을 마치기 전에 언급 할 마지막 용어는 0 max (0,-) max (0,-) 함수의 임계 값을 ** 힌지 손실 **이라고합니다. 위반 한 마진에 더 강력하게 페널티를 부과하는 max (0, −) 2max (0, −) 2 형식을 사용하는 제곱 힌지 손실 SVM (또는 L2-SVM)을 사용하는 대신 사람들에 대해 듣게 될 것입니다 (선형 대신 4 차). 제곱되지 않은 버전이 더 표준이지만 일부 데이터 세트에서는 제곱 힌지 손실이 더 잘 작동 할 수 있습니다. 이것은 교차 검증 중에 확인할 수 있습니다.

> 손실 함수는 훈련 세트에 대한 예측에 대한 우리의 불행을 정량화합니다.

Multiclass Support Vector Machine은 올바른 클래스의 점수가 다른 모든 점수보다 적어도 델타의 여유가 있기를 "원합니다". 어떤 클래스라도 빨간색 영역 (또는 그 이상) 내에 점수가 있으면 누적 손실이 발생합니다. 그렇지 않으면 손실이 0이 됩니다. 우리의 목표는 훈련 데이터의 모든 예제에 대해 이 제약 조건을 동시에 충족하고 가능한 한 낮은 총 손실을 제공 할 가중치를 찾는 것입니다.


==> MSVM은 올바른 클래스의 점수가 다른 모든 점수보다 적어도 델타의 여유가 있기를 바라는 모델입니다. 어떤 클래스라도 마진 내에 점수가 있으면 '누적 손실'이 발생합니다. 이 모델의 목표는 훈련 데이터의 모든 예제에 대해 제약 조건을 충족하면서, ** 가능한 낮은 총 손실 **을 제공할 '가중치'를 찾는 것입니다.





### SVM 관련 개념 추가 정리

**서포트 벡터 머신**(support vector machine, **SVM**[[1\]](https://ko.wikipedia.org/wiki/서포트_벡터_머신#cite_note-CorinnaCortes-1).[[2\]](https://ko.wikipedia.org/wiki/서포트_벡터_머신#cite_note-2))은 [기계 학습](https://ko.wikipedia.org/wiki/기계_학습)의 분야 중 하나로 패턴 인식, 자료 분석을 위한 [지도 학습](https://ko.wikipedia.org/wiki/지도_학습) 모델이며, 주로 [분류](https://ko.wikipedia.org/wiki/분류)와 [회귀 분석](https://ko.wikipedia.org/wiki/회귀_분석)을 위해 사용한다. 두 카테고리 중 어느 하나에 속한 데이터의 집합이 주어졌을 때, SVM 알고리즘은 주어진 데이터 집합을 바탕으로 하여 새로운 데이터가 어느 카테고리에 속할지 판단하는 비[확률적](https://ko.wikipedia.org/wiki/확률) 이진 [선형 분류](https://ko.wikipedia.org/wiki/선형_분류) 모델을 만든다. 만들어진 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데 SVM 알고리즘은 그 중 가장 큰 폭을 가진 경계를 찾는 알고리즘이다. SVM은 선형 분류와 더불어 비선형 분류에서도 사용될 수 있다. 비선형 분류를 하기 위해서 주어진 데이터를 고차원 특징 공간으로 사상하는 작업이 필요한데, 이를 효율적으로 하기 위해 [커널 트릭](https://ko.wikipedia.org/w/index.php?title=커널_트릭&action=edit&redlink=1)을 사용하기도 한다.



일반적으로, 서포트 벡터 머신은 분류 또는 회귀 분석에 사용 가능한 [초평면](https://ko.wikipedia.org/wiki/초평면)([영어](https://ko.wikipedia.org/wiki/영어): hyperplane) 또는 초평면들의 집합으로 구성되어 있다. 직관적으로, 초평면이 가장 가까운 학습 데이터 점과 큰 차이를 가지고 있으면 분류 오차([영어](https://ko.wikipedia.org/wiki/영어): classifier error)가 작기 때문에 좋은 분류를 위해서는 어떤 분류된 점에 대해서 가장 가까운 학습 데이터와 가장 먼 거리를 가지는 초평면을 찾아야 한다. 일반적으로 초기의 문제가 유한 차원 공간에서 다루어지는데, 종종 데이터가 [선형 구분](https://ko.wikipedia.org/wiki/선형_구분_가능)이 되지 않는 문제가 발생한다. 이러한 문제를 해결하기 위해 초기 문제의 유한 차원에서 더 높은 차원으로 대응시켜 분리를 쉽게 하는 방법이 제안되었다. 그 과정에서 계산량이 늘어나는 것을 막기 위해서, 각 문제에 적절한 [커널 함수](https://ko.wikipedia.org/w/index.php?title=커널_함수&action=edit&redlink=1) {\displaystyle k(x,y)}![k(x,y)](https://wikimedia.org/api/rest_v1/media/math/render/svg/7d18e060406f195657b2151490ca3d491f7a7ce0)를 정의한 SVM 구조를 설계하여 [내적 연산](https://ko.wikipedia.org/wiki/점곱)을 초기 문제의 변수들을 사용해서 효과적으로 계산할 수 있도록 한다.[[3\]](https://ko.wikipedia.org/wiki/서포트_벡터_머신#cite_note-3) 높은 차원 공간의 초평면은 점들의 집합과 상수 벡터의 내적 연산으로 정의된다. 초평면에 정의된 벡터들은 데이터 베이스 안에 나타나는 이미지 벡터 매개 변수들과의 선형적 결합이 되도록 선택된다. 이 선택된 초평면에서, 초평면에 대응된 점 {\displaystyle x}![x](https://wikimedia.org/api/rest_v1/media/math/render/svg/87f9e315fd7e2ba406057a97300593c4802b53e4)는 다음과 같은 관계가 성립한다.

만약 {\displaystyle k(x,y)}![k(x,y)](https://wikimedia.org/api/rest_v1/media/math/render/svg/7d18e060406f195657b2151490ca3d491f7a7ce0)가 {\displaystyle x}![x](https://wikimedia.org/api/rest_v1/media/math/render/svg/87f9e315fd7e2ba406057a97300593c4802b53e4) 와 {\displaystyle y}![y](https://wikimedia.org/api/rest_v1/media/math/render/svg/b8a6208ec717213d4317e666f1ae872e00620a0d)가 점점 멀어질 수록 작아진다면, 각각의 합은 테스트 점 {\displaystyle x}![x](https://wikimedia.org/api/rest_v1/media/math/render/svg/87f9e315fd7e2ba406057a97300593c4802b53e4)와 그와 대응되는 데이터 점{\displaystyle x_{i}}![x_i](https://wikimedia.org/api/rest_v1/media/math/render/svg/e87000dd6142b81d041896a30fe58f0c3acb2158)의 근접성의 정도를 나타내게 된다. 이러한 방식으로, 위 커널식의 합은 구별하고 싶은 집합안에 있는 데이터 점과 테스트 점간의 상대적인 근접성을 측정하는데 사용될 수 있다. 초기 공간에서 볼록하지 않는 집합안의 점 {\displaystyle x}![x](https://wikimedia.org/api/rest_v1/media/math/render/svg/87f9e315fd7e2ba406057a97300593c4802b53e4)가 높은 차원으로 대응되었을 때 오히려 더 복잡하고 어려워질 수도 있는데 이런 부분을 주의해야 한다.



<br>

## SVM 서포트 벡터 머신

1. 마진을 최대화 한다. 
2. 손실을 최소화 한다. 
3. 마진 안 혹은 데이터 분류 중 이상치가 속해있을 경우 파라미터(C) 값 조정을 통해 마진의 크기를 최적화한다.

<br>

## Kernel SVM

1. 저차원에서 직선이 나눠지지 않는 않는 경우 고차원에서 확인해본다.
2. <img width="598" alt="Screen Shot 2021-04-12 at 6 12 37 PM" src="https://user-images.githubusercontent.com/57430754/114380668-48f7e400-9bc5-11eb-9236-5411c95b3da2.png">

<img width="360" alt="Screen Shot 2021-04-12 at 6 12 45 PM" src="https://user-images.githubusercontent.com/57430754/114380678-4ac1a780-9bc5-11eb-9cb1-9d2af7ea1e5b.png">

사진과 같이 고차원에서 직선으로 나눠지는 경우 확인. (SVM은 직선 분류만 가능)

3. 고차원 상에서 직선으로 분류되었지만, 다시 저차원 상에서 확인해보면 곡선처럼 보인다. 그렇지만 이는 직선으로 분류한 것임.

<br>

## Multiclass SVM

1. 1대 all 분류로 볼 수 있음.
2. a, b, c가 있으면, a: b, c 등으로 볼 수 있음