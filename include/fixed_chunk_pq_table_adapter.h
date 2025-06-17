// fixed_chunk_pq_table_adapter.h
#pragma once
#include "pq_table_base.h"
#include "pq.h" // 包含原始FixedChunkPQTable定义

class FixedChunkPQTableAdapter : public PQTableBase {
  public:
    FixedChunkPQTableAdapter() {} // 初始化参数可能需要调整

    void train(idx_t n, const float* x) override {
        //FixedChunkPQTable write trained data to file and load into memory when search
    }

    void preprocess_query(float* query) override {
        pq_table.preprocess_query(query); // 假设原接口兼容
    }

    void populate_chunk_distances(const float* query, float* out_dists) override {
        pq_table.populate_chunk_distances(query, out_dists);
    }

    uint64_t get_num_chunks() override {
        return pq_table.get_num_chunks(); // 需要添加该方法到FixedChunkPQTable
    }

    // 扩展功能实现
    void apply_rotation(float* vec) const override {
//        if (pq_table.rotation_applied()) { // 假设存在此方法
//            pq_table.rotate(vec);
//        }
    }

    void compute_dists (const uint32_t *ids, const uint64_t n_ids, float *dists_out,
                       uint8_t *data, uint8_t *pq_coord_scratch, float* pq_dists) override {
        // this->data 所有的编码的pq code [1,2,3,4] [1,4,2,4] ......
        const auto chunks = this->get_num_chunks();
        diskann::aggregate_coords(ids, n_ids, data, chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, chunks, pq_dists, dists_out);
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

#ifdef EXEC_ENV_OLS
    void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) override
    {
        pq_table.load_pq_centroid_bin(files, pq_table_file, num_chunks);
    }
#else
    void load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) override
    {
        pq_table.load_pq_centroid_bin(pq_table_file, num_chunks);
    }
#endif

  private:
    diskann::FixedChunkPQTable pq_table;
    size_t npt;
    size_t ndim;
};
