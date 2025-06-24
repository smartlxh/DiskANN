#include "rabitq_quantizer.h"


//TODO: simd optimize
float RabitqQuantizer::fvec_L2sqr(const float* x, const float* y, size_t d) {
    size_t i;
    float res = 0;
    for (i = 0; i < d; i++) {
        const float tmp = x[i] - y[i];
        res += tmp * tmp;
    }
    return res;
}

float RabitqQuantizer::fvec_norm_L2sqr(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}

void RabitqQuantizer::train(size_t n, const float* x, const std::string data_file) {
    // read full-precision vector
    auto base_data_file = "data/sift/sift_learn.fbin";
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(base_data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    size_t dim = basedim32;
    npt = num_points = 10;
    ndim = dim;

    size_t BLOCK_SIZE = 200000;
    size_t block_size = num_points <= BLOCK_SIZE ? num_points : BLOCK_SIZE;
    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    std::vector<float> temp_centroid(d, 0);
    // todo template T[]
    std::unique_ptr<float[]> block_data_T = std::make_unique<float[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_tmp = std::make_unique<float[]>(block_size * dim);

    float* first_point = new float[dim];
    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        base_reader.read((char *)(block_data_tmp.get()), sizeof(float) * (cur_blk_size * dim));
        diskann::convert_types<float, float>(block_data_tmp.get(), block_data_T.get(), cur_blk_size, dim);

        //diskann::cout << "Processing points  [" << start_id << ", " << end_id << ").." << std::flush;
        for (size_t p = 0; p < cur_blk_size; p++)
        {
            float distance = 0;
            for (uint64_t d = 0; d < dim; d++)
            {
                temp_centroid[d] += block_data_T[p * dim + d];
                if (std::isnan(block_data_T[p * dim + d])) {
                    diskann::cout << "bbq2 nan " << p << " " << d << std::endl;
                }
                if (p == 0) {
                    first_point[d] = block_data_T[p * dim + d];
                } else {
                    distance += (block_data_T[p * dim + d] - first_point[d]) * (block_data_T[p * dim + d] - first_point[d]);
                }
            }

            diskann::cout << "distance " << p << " " << distance << std::endl;

            for (uint64_t d = 0; d < dim; d++) {
                diskann::cout << "point " << p <<":" << block_data_T[p * dim + d] << std::endl;
            }

        }
    }

    if (npt != 0) {
        for (size_t j = 0; j < d; j++) {
            temp_centroid[j] /= (float)npt;
            diskann::cout << "norlize " << j << " " << temp_centroid[j] << std::endl;
        }
    }

    center = std::move(temp_centroid);
    centroid = center.data();

    codes = new uint8_t[npt * code_size];
    compute_codes(block_data_T.get(), codes, npt);

    diskann::cout << "train done and print code" << std::endl;
    preprocess_query(first_point);
    for (int i = 0; i < npt; i++) {
        diskann::cout << "point " << i << " " << distance_to_code(codes + i * code_size) << std::endl;
    }
}

void RabitqQuantizer::preprocess_query(float* x) {
    if (_qb == 0) {
        // RaBitDistanceComputerNotQ
        // compute the distance from the query to the centroid
        if (centroid != nullptr) {
            query_fac.qr_to_c_L2sqr = fvec_L2sqr(x, centroid, d);
        } else {
            query_fac.qr_to_c_L2sqr = fvec_norm_L2sqr(x, d);
        }

        // subtract c, obtain P^(-1)(qr - c)
        rotated_q.resize(d);
        for (size_t i = 0; i < d; i++) {
            rotated_q[i] = x[i] - ((centroid == nullptr) ? 0 : centroid[i]);
        }

        // compute some numbers
        const float inv_d = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

        // do not quantize the query
        float sum_q = 0;
        for (size_t i = 0; i < d; i++) {
            sum_q += rotated_q[i];
        }

        query_fac.c1 = 2 * inv_d;
        query_fac.c2 = 0;
        query_fac.c34 = sum_q * inv_d;

        if (metric_type == diskann::Metric::INNER_PRODUCT) {
            // precompute if needed
            query_fac.qr_norm_L2sqr = fvec_norm_L2sqr(x, d);
        }

    } else {
        // compute the distance from the query to the centroid
        if (centroid != nullptr) {
            query_fac.qr_to_c_L2sqr = fvec_L2sqr(x, centroid, d);
        } else {
            query_fac.qr_to_c_L2sqr = fvec_norm_L2sqr(x, d);
        }

        // allocate space
        rotated_qq.resize(d);

        // rotate the query
        std::vector<float> rotated_q(d);
        for (size_t i = 0; i < d; i++) {
            rotated_q[i] = x[i] - ((centroid == nullptr) ? 0 : centroid[i]);
        }

        // compute some numbers
        const float inv_d = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

        // quantize the query. compute min and max
        float v_min = std::numeric_limits<float>::max();
        float v_max = std::numeric_limits<float>::lowest();
        for (size_t i = 0; i < d; i++) {
            const float v_q = rotated_q[i];
            v_min = std::min(v_min, v_q);
            v_max = std::max(v_max, v_q);
        }

        const float pow_2_qb = 1 << _qb;

        const float delta = (v_max - v_min) / (pow_2_qb - 1);
        const float inv_delta = 1.0f / delta;

        size_t sum_qq = 0;
        for (int32_t i = 0; i < d; i++) {
            const float v_q = rotated_q[i];

            // a default non-randomized SQ
            const int v_qq = std::round((v_q - v_min) * inv_delta);

            rotated_qq[i] = std::min(255, std::max(0, v_qq));
            sum_qq += v_qq;
        }

        // rearrange the query vector
        popcount_aligned_dim = ((d + 7) / 8) * 8;
        size_t offset = (d + 7) / 8;

        rearranged_rotated_qq.resize(offset * _qb);
        std::fill(rearranged_rotated_qq.begin(), rearranged_rotated_qq.end(), 0);

        for (size_t idim = 0; idim < d; idim++) {
            for (size_t iv = 0; iv < _qb; iv++) {
                const bool bit = ((rotated_qq[idim] & (1 << iv)) != 0);
                rearranged_rotated_qq[iv * offset + idim / 8] |=
                    bit ? (1 << (idim % 8)) : 0;
            }
        }

        query_fac.c1 = 2 * delta * inv_d;
        query_fac.c2 = 2 * v_min * inv_d;
        query_fac.c34 = inv_d * (delta * sum_qq + d * v_min);

        if (metric_type == diskann::Metric::INNER_PRODUCT) {
            // precompute if needed
            query_fac.qr_norm_L2sqr = fvec_norm_L2sqr(x, d);
        }
    }
}

void RabitqQuantizer::load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data) {
    // TODO
    // load the PQ compressed vectors generated in `train`
    // for now, we just keep the vector in memory to avoid the step write and load.
    train(0, nullptr, bin_file);
}

void RabitqQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
{
    compute_codes_core(x, codes, n, centroid);
}

void RabitqQuantizer::compute_codes_core(
    const float* x,
    uint8_t* codes,
    size_t n,
    const float* centroid_in) const {
//    FAISS_ASSERT(codes != nullptr);
//    FAISS_ASSERT(x != nullptr);
//    FAISS_ASSERT(
//        (metric_type == Metric::L2 ||
//         metric_type == MetricType::INNER_PRODUCT));

    if (n == 0) {
        return;
    }

    // compute some helper constants
    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    // compute codes
#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < n; i++) {
        // ||or - c||^2
        float norm_L2sqr = 0;
        // ||or||^2, which is equal to ||P(or)||^2 and ||P^(-1)(or)||^2
        float or_L2sqr = 0;
        // dot product
        float dp_oO = 0;

        // the code
        uint8_t* code = codes + i * code_size;
        FactorsData* fac = reinterpret_cast<FactorsData*>(code + (d + 7) / 8);

        // cleanup it
        if (code != nullptr) {
            memset(code, 0, code_size);
        }

        for (size_t j = 0; j < d; j++) {
            const float or_minus_c = x[i * d + j] -
                                     ((centroid_in == nullptr) ? 0 : centroid_in[j]);
            norm_L2sqr += or_minus_c * or_minus_c;
            or_L2sqr += x[i * d + j] * x[i * d + j];

            const bool xb = (or_minus_c > 0);

            dp_oO += xb ? or_minus_c : (-or_minus_c);

            // store the output data
            if (code != nullptr) {
                if (xb) {
                    // enable a particular bit
                    code[j / 8] |= (1 << (j % 8));
                    //diskann::cout << "codes: " << j << " " << code[j / 8] << std::endl;
                }
            }
        }

        // compute factors

        // compute the inverse norm
        const float inv_norm_L2 =
            (std::abs(norm_L2sqr) < std::numeric_limits<float>::epsilon())
                ? 1.0f
                : (1.0f / std::sqrt(norm_L2sqr));
        dp_oO *= inv_norm_L2;
        dp_oO *= inv_d_sqrt;

        const float inv_dp_oO =
            (std::abs(dp_oO) < std::numeric_limits<float>::epsilon())
                ? 1.0f
                : (1.0f / dp_oO);

        fac->or_minus_c_l2sqr = norm_L2sqr;
        if (metric_type == diskann::Metric::INNER_PRODUCT) {
            fac->or_minus_c_l2sqr -= or_L2sqr;
        }

        fac->dp_multiplier = inv_dp_oO * std::sqrt(norm_L2sqr);
    }
}

void RabitqQuantizer::compute_dists(const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                   uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists) {
    memset(dists_out, 0, n_ids * sizeof(float));
    for (size_t i = 0; i < n_ids; i++) {
        uint8_t *code = codes + ids[i] * code_size;
        auto distance = distance_to_code(code);
        dists_out[i] = distance;
        //diskann::cout << "dists_out: " << i << " " << distance << std::endl;
    }
}

float RabitqQuantizer::distance_to_code(const uint8_t* code) {
    if (_qb == 0) {
        // RaBitDistanceComputerNotQ distance_to_code
        // split the code into parts
        const uint8_t* binary_data = code;
        const FactorsData* fac =
            reinterpret_cast<const FactorsData*>(code + (d + 7) / 8);


        // this is the baseline code
        //
        // compute <q,o> using floats
        float dot_qo = 0;
        // It was a willful decision (after the discussion) to not to pre-cache
        //   the sum of all bits, just in order to reduce the overhead per vector.
        uint64_t sum_q = 0;
        for (size_t i = 0; i < d; i++) {
            // extract i-th bit
            const uint8_t masker = (1 << (i % 8));
            const bool b_bit = ((binary_data[i / 8] & masker) == masker);

            // accumulate dp
            dot_qo += (b_bit) ? rotated_q[i] : 0;
            // accumulate sum-of-bits
            sum_q += (b_bit) ? 1 : 0;
        }

        float final_dot = 0;
        // dot-product itself
        final_dot += query_fac.c1 * dot_qo;
        // normalizer coefficients
        final_dot += query_fac.c2 * sum_q;
        // normalizer coefficients
        final_dot -= query_fac.c34;

        // this is ||or - c||^2 - (IP ? ||or||^2 : 0)
        const float or_c_l2sqr = fac->or_minus_c_l2sqr;

        // pre_dist = ||or - c||^2 + ||qr - c||^2 -
        //     2 * ||or - c|| * ||qr - c|| * <q,o> - (IP ? ||or||^2 : 0)
        const float pre_dist = or_c_l2sqr + query_fac.qr_to_c_L2sqr -
                               2 * fac->dp_multiplier * final_dot;

        if (metric_type == diskann::Metric::L2) {
            // ||or - q||^ 2
            return pre_dist;
        } else {
            // metric == MetricType::METRIC_INNER_PRODUCT

            // this is ||q||^2
            const float query_norm_sqr = query_fac.qr_norm_L2sqr;

            // 2 * (or, q) = (||or - q||^2 - ||q||^2 - ||or||^2)
            return -0.5f * (pre_dist - query_norm_sqr);
        }

    } else {
        // RaBitDistanceComputerQ distance_to_code
        // split the code into parts
        const uint8_t* binary_data = code;
        const FactorsData* fac =
            reinterpret_cast<const FactorsData*>(code + (d + 7) / 8);

        // // this is the baseline code
        // //
        // // compute <q,o> using integers
        // size_t dot_qo = 0;
        // for (size_t i = 0; i < d; i++) {
        //     // extract i-th bit
        //     const uint8_t masker = (1 << (i % 8));
        //     const uint8_t bit = ((binary_data[i / 8] & masker) == masker) ? 1 :
        //     0;
        //
        //     // accumulate dp
        //     dot_qo += bit * rotated_qq[i];
        // }

        // this is the scheme for popcount
        const size_t di_8b = (d + 7) / 8;
        const size_t di_64b = (di_8b / 8) * 8;

        uint64_t dot_qo = 0;
        for (size_t j = 0; j < _qb; j++) {
            const uint8_t* query_j = rearranged_rotated_qq.data() + j * di_8b;

            // process 64-bit popcounts
            uint64_t count_dot = 0;
            for (size_t i = 0; i < di_64b; i += 8) {
                const auto qv = *(const uint64_t*)(query_j + i);
                const auto yv = *(const uint64_t*)(binary_data + i);
                count_dot += __builtin_popcountll(qv & yv);
            }

            // process leftovers
            for (size_t i = di_64b; i < di_8b; i++) {
                const auto qv = *(query_j + i);
                const auto yv = *(binary_data + i);
                count_dot += __builtin_popcount(qv & yv);
            }

            dot_qo += (count_dot << j);
        }

        // It was a willful decision (after the discussion) to not to pre-cache
        //   the sum of all bits, just in order to reduce the overhead per vector.
        uint64_t sum_q = 0;
        {
            // process 64-bit popcounts
            for (size_t i = 0; i < di_64b; i += 8) {
                const auto yv = *(const uint64_t*)(binary_data + i);
                sum_q += __builtin_popcountll(yv);
            }

            // process leftovers
            for (size_t i = di_64b; i < di_8b; i++) {
                const auto yv = *(binary_data + i);
                sum_q += __builtin_popcount(yv);
            }
        }

        float final_dot = 0;
        // dot-product itself
        final_dot += query_fac.c1 * dot_qo;
        // normalizer coefficients
        final_dot += query_fac.c2 * sum_q;
        // normalizer coefficients
        final_dot -= query_fac.c34;

        // this is ||or - c||^2 - (IP ? ||or||^2 : 0)
        const float or_c_l2sqr = fac->or_minus_c_l2sqr;

        // pre_dist = ||or - c||^2 + ||qr - c||^2 -
        //     2 * ||or - c|| * ||qr - c|| * <q,o> - (IP ? ||or||^2 : 0)
        const float pre_dist = or_c_l2sqr + query_fac.qr_to_c_L2sqr -
                               2 * fac->dp_multiplier * final_dot;

        if (metric_type == diskann::Metric::L2) {
            // ||or - q||^ 2
            return pre_dist;
        } else {
            // metric == MetricType::METRIC_INNER_PRODUCT

            // this is ||q||^2
            const float query_norm_sqr = query_fac.qr_norm_L2sqr;

            // 2 * (or, q) = (||or - q||^2 - ||q||^2 - ||or||^2)
            return -0.5f * (pre_dist - query_norm_sqr);
        }
    }
}