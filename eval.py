from improved_gan.inception_score.test_inception import test_inception
from TTUR.precalc_stats_example import precalc_stats
from TTUR.fid_example import calc_fid
import argparse

def eval_all(real_path, gen_path, save_file):
    print("Getting Inception Score")
    inception_score_mean, inception_score_std = test_inception(gen_path)
    print("Getting Frechet Inception Distance")
    mu, sigma = precalc_stats(data_path=real_path, save_stats=False)
    fid_score = calc_fid(image_path=gen_path, mu_real=mu, sigma_real=sigma)
    if save_file:
        with open(save_file, "w") as f:
            f.write("Inception Score Mean: %f \n" % inception_score_mean)
            f.write("Inception Score Std: %f \n" % inception_score_std)
            f.write("FID Score: %f \n" % fid_score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", required=True, type=str, help="Path to list of real png images.  Needed to compute FID")
    parser.add_argument("--gen_path", required=True, type=str, help="Path to list of generated png images.  Needed to compute both FID and Inception score")
    parser.add_argument("--save_file", required=False, type=str, help="File to save scores in.  If unspecified, the scores will not be saved.")
    args = parser.parse_args()
    eval_all(args.real_path, args.gen_path, args.save_file)
