// pq_table_base.h
#pragma once
#include <cstdint>
#include <vector>

template <typename T>
class PQTableBase {
  public:
    virtual ~PQTableBase() = default;

    // 查询预处理（中心化/旋转）
    virtual void preprocess_query(const T* query) = 0;

    // 计算每个chunk的距离表
    virtual void populate_chunk_distances(const T* query, float* out_dists) const = 0;

    // 聚合坐标（根据ID获取PQ码）
    virtual void aggregate_coords(const uint32_t* ids, uint64_t n_ids,
                                  const uint8_t* pq_data, uint64_t n_chunks,
                                  uint8_t* out_coords) const = 0;

    // 获取chunk数量 pq
    virtual uint64_t get_num_chunks() const = 0;

    // 支持OPQ等扩展功能的虚函数
    virtual void apply_rotation(T* vec) const { /* 默认无旋转 */ }

    virtual void compute_dists (const uint32_t *ids, const uint64_t n_ids, float *dists_out);

    virtual uint64_t get_num_points();

    virtual void load_pq_compressed_vectors(const std::string &bin_file, uint8_t* &data);

#ifdef EXEC_ENV_OLS
    virtual void load_pq_centroid_bin(MemoryMappedFiles &files, const char *pq_table_file, size_t num_chunks) {};
#else
    virtual load_pq_centroid_bin(const char *pq_table_file, size_t num_chunks) {};
#endif
};
