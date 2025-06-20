// fixed_chunk_pq_table_adapter.h
#pragma once
#include "pq_table_base.h"
#include "pq.h" // 包含原始FixedChunkPQTable定义
#include "distance.h"

struct FactorsData {
    // ||or - c||^2 - ((metric==IP) ? ||or||^2 : 0)
    float or_minus_c_l2sqr = 0;
    float dp_multiplier = 0;
};

struct QueryFactorsData {
    float c1 = 0;
    float c2 = 0;
    float c34 = 0;

    float qr_to_c_L2sqr = 0;
    float qr_norm_L2sqr = 0;
};

// the code copied from faiss#RabitQuantizer.cpp
class RabitqQuantizer : public PQTableBase {
  public:
    RabitqQuantizer(size_t d = 0, diskann::Metric metric = diskann::Metric::L2) {
        this->d = d;
        this->metric_type = metric;
        code_size = get_code_size(d);
    }

    static size_t get_code_size(const size_t d) {
        return (d + 7) / 8 + sizeof(FactorsData);
    }

    void train(size_t n, const float* x, const std::string data_file) override;

    // every vector is expected to take (d + 7) / 8 + sizeof(FactorsData) bytes,
    void compute_codes(const float* x, uint8_t* codes, size_t n);

    void preprocess_query(float* x) override;

    void populate_chunk_distances(const float* query, float* out_dists) override {
        // do nothing
    }

    uint64_t get_num_chunks() override {
        return 0;
    }

    // 扩展功能实现
    void apply_rotation(float* vec) const override {
        //        if (pq_table.rotation_applied()) { // 假设存在此方法
        //            pq_table.rotate(vec);
        //        }
    }

    void compute_dists (const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists) override;




    uint64_t get_num_points() override {
        return npt;
    }

    void load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data) override;

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) override
    {
    }
#else
    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) override
    {
    }
#endif

    private:
      void compute_codes_core(
          const float* x,
          uint8_t* codes,
          size_t n,
          const float* centroid_in) const;

      float distance_to_code(const uint8_t* code);
      float fvec_L2sqr(const float* x, const float* y, size_t d);
      float fvec_norm_L2sqr(const float* x, size_t d);

  private:
    // ---------------------------
    // rabitq
    // center of all points
    std::vector<float> center;
    float* centroid = nullptr;

    int d;        ///< vector dimension

    // --------------------------- RaBitDistanceComputerNotQ
    // the rotated and quantized query (qr - c)
    std::vector<uint8_t> rotated_q;

    // --------------------------- RaBitDistanceComputerQ
    // the rotated and quantized query (qr - c)
    std::vector<uint8_t> rotated_qq;
    // the smallest value divisible by 8 that is not smaller than dim
    size_t popcount_aligned_dim = 0;

    // we're using the proposed relayout-ed scheme from 3.3 that allows
    // using popcounts for computing the distance.
    std::vector<uint8_t> rearranged_rotated_qq;
    QueryFactorsData query_fac;
    uint8_t* codes;
    size_t code_size;
    diskann::Metric metric_type;
    // ---------------------------
    // pq
    uint8_t _qb = 0;
    size_t npt;
    size_t ndim;
};
