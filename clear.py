import os
import time
import glob

def reset_csv_file_metadata(results_dir="results"):
    """
    递归重置指定目录及其子目录下所有 CSV 文件的元数据（修改时间、访问时间等）
    
    Args:
        results_dir: 要处理的根目录
    """
    # 检查目录是否存在
    if not os.path.exists(results_dir) or not os.path.isdir(results_dir):
        print(f"错误: '{results_dir}' 目录不存在")
        return False
    
    # 获取当前时间戳
    current_time = time.time()
    
    # 统计处理文件数量
    processed_count = 0
    
    # 递归遍历目录和子目录
    for root, dirs, files in os.walk(results_dir):
        # 筛选出 CSV 文件
        csv_files = [f for f in files if f.lower().endswith('.csv')]
        
        for csv_file in csv_files:
            file_path = os.path.join(root, csv_file)
            try:
                # 修改文件的访问时间和修改时间
                os.utime(file_path, (current_time, current_time))
                print(f"已重置文件的时间属性: {file_path}")
                processed_count += 1
            except Exception as e:
                print(f"处理文件 '{file_path}' 时出错: {str(e)}")
    
    if processed_count > 0:
        print(f"完成! 已处理 {processed_count} 个 CSV 文件")
    else:
        print(f"警告: 在 '{results_dir}' 及其子目录中没有找到任何 CSV 文件")
    
    return processed_count > 0

if __name__ == "__main__":
    # 调用函数处理 results 目录及其所有子目录下的 CSV 文件
    reset_csv_file_metadata()