module WJFCVNN
using ColorTypes, FixedPointNumbers
using Flux, Metalhead
using Random
export loadCIFAR10, makeBatches
function loadCIFAR10(path::String)::Dict{String,Vector{Matrix{N0f8}}}
    classNames = String[]
    open(joinpath(path, "batches.meta.txt"), "r") do fileMeta
        for i in 1:10
            line = readline(fileMeta)
            push!(classNames, line)
        end
    end
    result = Dict{String,Vector{Matrix{N0f8}}}()
    for idxClass in 1:10
        result[classNames[idxClass]] = Vector{Matrix{N0f8}}()
    end
    for idxBin in 1:5
        open(joinpath(path, "data_batch_$idxBin.bin"), "r") do fileBin
            for idxBlock in 1:10000
                classID = read(fileBin, UInt8)
                data = Matrix{N0f8}(undef, (32,96))
                read!(fileBin, data)
                className = classNames[classID + 1]
                push!(result[className], data)
            end
        end
    end
    return result
end

function getImage(data::Matrix{N0f8})::Matrix{RGB{N0f8}}
    (m,n) = size(data)
    @assert n % 3 == 0
    n ÷= 3
    result = Matrix{RGB{N0f8}}(undef, (n, m))
    for row in 1:m
        for col in 1:n
            result[col, row] = RGB{N0f8}(
                data[row, col], data[row, n + col], data[row, n + n + col],
                )
        end
    end
    return result
end

function makeBatches(
    labelledData::Dict{String,Vector{Matrix{N0f8}}},
    batchSize::Int
    )::Vector{Tuple{Array{Float32, 4}, Matrix{Float32}}}
    indexKeySet = keys(labelledData)
    indexKeys = [key for key in indexKeySet]
    sort!(indexKeys)
    keyIDs = Dict{String, UInt8}(
        indexKeys[i]=>UInt8(i) for i in 1:length(indexKeys))
    indices = Vector{Tuple{String,Int}}()
    for (className, classData) in labelledData
        for i in 1:length(classData)
            push!(indices, (className, i))
        end
    end
    shuffle!(indices)
    numBatches = (length(indices) + batchSize - 1) ÷ batchSize
    (m, n) = size(labelledData[indices[1][1]][indices[1][2]])
    @assert n % 3 == 0
    n ÷= 3
    result = Vector{Tuple{Array{Float32, 4}, Matrix{Float32}}}()
    for i in 1:batchSize:length(indices)
        batchIndices = if i + batchSize - 1 > length(indices)
            indices[length(indices) - batchSize + 1:length(indices)]
        else
            indices[i:i + batchSize - 1]
        end
        batchData = Array{Float32, 4}(undef, (m, n, 3, batchSize))
        batchLabelIDs = fill(Float32(0), (10,batchSize))
        for j in 1:batchSize
            bij = batchIndices[j]
            data = labelledData[bij[1]][bij[2]]
            for x in 1:m
                for y in 1:n
                    batchData[x, y, 1, j] = data[x, y]
                    batchData[x, y, 2, j] = data[x, y + n]
                    batchData[x, y, 3, j] = data[x, y + n + n]
                end
            end
            batchLabelIDs[keyIDs[bij[1]], j] = Float32(1)
        end
        push!(result, (batchData, batchLabelIDs))
    end
    return result
end

end # module
