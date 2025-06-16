// fixed_chunk_pq_table_adapter.h
#pragma once
#include "pq_table_base.h"
#include "pq.h" // 包含原始FixedChunkPQTable定义

template <typename T>
class FixedChunkPQTableAdapter : public PQTableBase<T> {
  public:
    FixedChunkPQTableAdapter() : pq_table(256) {} // 初始化参数可能需要调整

    void preprocess_query(const T* query) override {
        pq_table.preprocess_query(query); // 假设原接口兼容
    }

    void populate_chunk_distances(const T* query, float* out_dists) const override {
        pq_table.populate_chunk_distances(query, out_dists);
    }

    void aggregate_coords(const uint32_t* ids, uint64_t n_ids,
                          const uint8_t* pq_data, uint64_t n_chunks,
                          uint8_t* out_coords) const override {
        diskann::aggregate_coords(ids, n_ids, pq_data, n_chunks, out_coords);
    }

    uint64_t get_num_chunks() const override {
        return pq_table.get_num_chunks(); // 需要添加该方法到FixedChunkPQTable
    }

    // 扩展功能实现
    void apply_rotation(T* vec) const override {
        if (pq_table.rotation_applied()) { // 假设存在此方法
            pq_table.rotate(vec);
        }
    }

    void compute_dists (const uint32_t *ids, const uint64_t n_ids, float *dists_out) override {
        // this->data 所有的编码的pq code [1,2,3,4] [1,4,2,4] ......
        diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
    }

    uint64_t get_num_points() override {
        return npt;
    }

    void load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data) override {
        #ifdef EXEC_ENV_OLS
            diskann::load_bin<uint8_t>(files, bin_file, data, npt, ndim); // load pq_compressed_vectors
        #else
            diskann::load_bin<uint8_t>(bin_file, data, npt, ndim);
        #endif
    }

  private:
    FixedChunkPQTable pq_table;
    private size_t npt;
    private size_t ndim;
};
