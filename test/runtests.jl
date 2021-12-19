import Pkg
Pkg.activate(".")
using WJFCVNN
using Flux, Metalhead
using JLD2
#@load "pretrained/ResNet18-CIFAR10.jld2" resnet
vgg = VGG((32, 32); config = Metalhead.vgg_config[:D],
                          inchannels = 3,
                          batchnorm = true,
                          nclasses = 10,
                          fcsize = 1024,
                          dropout = 0.5)
vgg = vgg |> gpu

cifar10 = loadCIFAR10("D:\\Julia\\data\\ImageDatasets\\cifar-10-batches-bin")
batchData = makeBatches(cifar10, 1024)
gpuBatchData = [(x|>gpu,y|>gpu) for (x,y) in batchData]
loss(x, y) = Flux.Losses.mse(vgg(x), y)
ps = Flux.params(vgg)
opt = ADAM()
numEpoches = 200

t1 = time()
for epoch = 1:numEpoches
    Flux.train!(loss, ps, gpuBatchData, opt)
    err = 0.0
    for (x,y) in gpuBatchData
        err += loss(x,y)
    end
    t2 = time()
    elapsed = t2 - t1
    println("epoch $epoch, elapsed time $elapsed: err = $err")
end
vgg = vgg |> cpu
@save "pretrained/VGG16-CIFAR10.jld2" vgg #resnet
