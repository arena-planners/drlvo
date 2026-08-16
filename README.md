# drlvo

Arena wrapper for the **DRL-VO** (Velocity Obstacle) navigator. Adapted from [TempleRAIL/drl_vo_nav](https://github.com/TempleRAIL/drl_vo_nav) (humble branch).

## Run

```sh
arena launch robot.mobile:=drl robot.mobile.planner:=drlvo
```

Requires a global plan. Defaults to `nav2/navfn`.

## Files

- `planner.py`: entry point. Scan + lookahead-on-path produces `[vx, wz]`.
- `policy.py`: vendored ResNet+PPO model loader.
- `planner.yaml`: observation manifest.
- `weights.yaml`: pulls `drl_vo.zip` from [arena-rosnav/drlvo](https://huggingface.co/arena-rosnav/drlvo) on `arena feature planners add drlvo`.
- See [ATTRIBUTION.md](ATTRIBUTION.md) for code provenance. Weight metadata lives on the HF repo.

## License

GPL-3.0 (inherited from upstream).
